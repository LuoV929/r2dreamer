"""Offline policy evaluation from saved agent checkpoints.

Runs the same env stepping loop as ``OnlineTrainer.eval`` (parallel eval envs,
``agent.act(..., eval=True)``), without training or replay buffer.

Examples
--------
Single checkpoint (default Hydra overrides match size100M Atari Breakout training)::

    python eval_checkpoint.py --checkpoints logdir/run/latest.pt --num-rounds 5 --seed 0

Several checkpoints (e.g. baseline + pruned)::

    python eval_checkpoint.py \\
        --checkpoints baseline/latest.pt pruned/reward_30pct_finetuned.pt \\
        --num-rounds 10

Extra Hydra overrides (must come after ``--``)::

    python eval_checkpoint.py -c ckpt.pt --num-rounds 3 -- --env.eval_episode_num=20

Pruned reward checkpoints: ``reward.mlp.layers.reward_linear0`` out_features are read
from the state dict and ``model.reward.units`` is set before building ``Dreamer``.

RSSM + ConvDecoder pruned (``prune_rssm_decoder`` / ``finetune_pruned_rssm_decoder``):
``rssm._obs_net.obs_net_0.weight`` 的第 0 维推断 ``H``，并设置 ``model.hidden``,
``model.rssm.hidden``, ``model.decoder.cnn.units`` 为 ``H``（**不**改 ``model.units``，
Actor/Encoder 等仍为原宽度）。

Encoder CNN 末层通道剪枝（``prune_encoder`` / ``finetune_pruned_encoder``）：从
``state_dict`` 中 ``encoder.encoders.0.layers.*`` 最后一层 4D 卷积权重的输出通道数推断
``last_out_channels``，并通过 ``open_dict`` 写入 ``model.encoder.cnn.last_out_channels``，
使 ``ConvEncoder`` 与 checkpoint 对齐。
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import re
import sys

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

import tools
from dreamer import Dreamer
from envs import make_envs


def _default_config_dir() -> str:
    return str(pathlib.Path(__file__).resolve().parent / "configs")


DEFAULT_OVERRIDES = [
    "env=atari",
    "model=size100M",
    "model.rep_loss=dreamer",
    "model.compile=False",
    "device=cuda:0",
]


@torch.no_grad()
def evaluate_once(
    agent,
    eval_envs,
    max_steps_per_episode: int | None = None,
) -> dict[str, float]:
    """One eval round: each parallel env runs until its first episode ends.

    If ``max_steps_per_episode`` is set, any env that exceeds this step count is
    treated as finished (``done`` forced True) so a single degenerate trajectory
    cannot dominate wall time and distort mean episode length.
    """
    envs = eval_envs
    agent.eval()
    done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
    once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
    steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
    returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)

    agent_state = agent.get_initial_state(envs.env_num)
    act = agent_state["prev_action"].clone()

    while not once_done.all():
        steps += (~done & ~once_done).to(torch.int32)
        act_cpu = act.detach().to("cpu")
        done_cpu = done.detach().to("cpu")
        trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
        trans = trans_cpu.to(agent.device, non_blocking=True)
        done = done_cpu.to(agent.device)

        if max_steps_per_episode is not None:
            over = (steps >= int(max_steps_per_episode)) & ~once_done
            done = done | over

        trans["action"] = act
        act, agent_state = agent.act(trans, agent_state, eval=True)
        returns += trans["reward"][:, 0] * ~once_done
        once_done |= done

    agent.train()
    rets = returns.detach().cpu()
    lens = steps.detach().cpu().to(torch.float32)
    return {
        "mean_return": float(rets.mean()),
        "std_return": float(rets.std(unbiased=False)),
        "min_return": float(rets.min()),
        "max_return": float(rets.max()),
        "mean_length": float(lens.mean()),
        "min_length": float(lens.min()),
        "max_length": float(lens.max()),
        "length_std": float(lens.std(unbiased=False)),
        "num_episodes": int(envs.env_num),
    }


def load_agent(
    checkpoint_path: pathlib.Path,
    cfg,
    obs_space,
    act_space,
    device: torch.device,
) -> Dreamer:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "agent_state_dict" not in checkpoint:
        raise KeyError(f"{checkpoint_path}: expected key 'agent_state_dict'")
    state_dict = checkpoint["agent_state_dict"]

    for key in list(state_dict.keys()):
        t = state_dict[key]
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            state_dict[key] = t.float()

    cfg_model = copy.deepcopy(cfg.model)

    # Encoder CNN 末层通道剪枝：与默认 depth*mults[-1] 不一致时写入 last_out_channels
    _last_layer_idx = -1
    _last_conv_c: int | None = None
    for _k, _t in state_dict.items():
        if not isinstance(_t, torch.Tensor) or _t.dim() != 4:
            continue
        _m = re.match(r"encoder\.encoders\.0\.layers\.(\d+)\.weight$", _k)
        if not _m:
            continue
        _li = int(_m.group(1))
        if _li > _last_layer_idx:
            _last_layer_idx = _li
            _last_conv_c = int(_t.shape[0])
    if _last_conv_c is not None:
        _ec = cfg_model.encoder.cnn
        _c_def = int(_ec.depth) * int(list(_ec.mults)[-1])
        if _last_conv_c != _c_def:
            with open_dict(_ec):
                _ec.last_out_channels = _last_conv_c

    wkey = "reward.mlp.layers.reward_linear0.weight"
    if wkey in state_dict:
        cfg_model.reward.units = int(state_dict[wkey].shape[0])

    wa, wv = "actor.mlp.layers.actor_linear0.weight", "value.mlp.layers.value_linear0.weight"
    wa2 = "actor.mlp.layers.actor_linear2.weight"
    if wa in state_dict and wv in state_dict:
        ua, uv = int(state_dict[wa].shape[0]), int(state_dict[wv].shape[0])
        if ua != uv:
            raise RuntimeError(f"Actor/Value first-hidden out mismatch: {ua} vs {uv}")
        if wa2 in state_dict and int(state_dict[wa2].shape[0]) != ua:
            raise RuntimeError(
                "Actor MLP 各层隐藏宽度不一致，无法用单一 cfg.model.actor.units 构建 Dreamer。"
                "请使用修正后的 prune_actor_critic 重新导出 checkpoint。"
            )
        cfg_model.actor.units = ua
        cfg_model.critic.units = uv

    # RSSM + CNN decoder 宽度剪枝：Hydra 默认 hidden 仍为 768，须与 checkpoint 对齐
    k_obs = "rssm._obs_net.obs_net_0.weight"
    if str(getattr(cfg_model, "rep_loss", "")) == "dreamer" and k_obs in state_dict:
        H_ckpt = int(state_dict[k_obs].shape[0])
        H_cfg = int(cfg_model.hidden) if hasattr(cfg_model, "hidden") else H_ckpt
        if H_ckpt != H_cfg:
            cfg_model.hidden = H_ckpt
            if hasattr(cfg_model, "rssm") and hasattr(cfg_model.rssm, "hidden"):
                cfg_model.rssm.hidden = H_ckpt
            if hasattr(cfg_model, "decoder") and hasattr(cfg_model.decoder, "cnn"):
                cfg_model.decoder.cnn.units = H_ckpt
            k_dyn = "rssm._deter_net._dyn_in0.0.weight"
            k_sp1 = "decoder._cnn.sp1.0.weight"
            if k_dyn in state_dict and int(state_dict[k_dyn].shape[0]) != H_ckpt:
                raise RuntimeError(
                    f"RSSM dyn_in0 输出维 {state_dict[k_dyn].shape[0]} 与 obs_net 首层 {H_ckpt} 不一致"
                )
            if k_sp1 in state_dict and int(state_dict[k_sp1].shape[0]) != 2 * H_ckpt:
                raise RuntimeError(
                    f"decoder._cnn.sp1[0] 输出维 {state_dict[k_sp1].shape[0]} 与 2*H={2 * H_ckpt} 不一致"
                )

    agent = Dreamer(cfg_model, obs_space, act_space).to(device)
    missing, unexpected = agent.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict: missing={missing}, unexpected={unexpected}")
    agent.float()
    return agent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline eval for Dreamer checkpoints.")
    parser.add_argument(
        "-c",
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more .pt files containing 'agent_state_dict'.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=_default_config_dir(),
        help="Directory that contains configs.yaml and env/ model/ subdirs.",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="How many times to repeat full parallel eval (total episodes = rounds * eval workers).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override cfg.device (e.g. cuda:0). Defaults to Hydra config.",
    )
    parser.add_argument(
        "--max-eval-steps",
        type=int,
        default=None,
        help="Force episode end after this many env steps per worker (Breakout default "
        "time_limit can be ~27k; set e.g. 5000 to cap degenerate long episodes).",
    )
    args, hydra_overrides = parser.parse_known_args(argv)
    if hydra_overrides and hydra_overrides[0] == "--":
        hydra_overrides = hydra_overrides[1:]

    overrides = list(DEFAULT_OVERRIDES)
    overrides.extend(hydra_overrides)

    config_dir = args.config_dir
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="configs", overrides=overrides)

    if args.device is not None:
        cfg.device = args.device

    tools.set_seed_everywhere(int(args.seed))
    device = torch.device(cfg.device)

    _, eval_envs, obs_space, act_space = make_envs(cfg.env)
    episodes_per_round = eval_envs.env_num

    print(f"device={device}, eval_parallel_envs={episodes_per_round}, num_rounds={args.num_rounds}")
    if args.max_eval_steps is not None:
        print(f"max_eval_steps (forced episode cap)={args.max_eval_steps}")
    print(f"total_episodes ≈ {episodes_per_round * args.num_rounds}")

    for ckpt_str in args.checkpoints:
        ckpt_path = pathlib.Path(ckpt_str).expanduser()
        if not ckpt_path.is_file():
            print(f"[skip] not a file: {ckpt_path}", file=sys.stderr)
            continue

        print(f"\n=== {ckpt_path} ===")
        agent = load_agent(ckpt_path, cfg, obs_space, act_space, device)

        all_means: list[float] = []
        for r in range(args.num_rounds):
            m = evaluate_once(agent, eval_envs, max_steps_per_episode=args.max_eval_steps)
            all_means.append(m["mean_return"])
            print(
                f"  round {r + 1}/{args.num_rounds}: "
                f"mean_return={m['mean_return']:.2f} "
                f"(std_episode={m['std_return']:.2f}, "
                f"len_mean={m['mean_length']:.1f}, len_min={m['min_length']:.0f}, "
                f"len_max={m['max_length']:.0f}, len_std={m['length_std']:.1f})"
            )

        stacked = torch.tensor(all_means)
        print(
            f"  aggregate over rounds: mean_of_round_means={stacked.mean().item():.2f}, "
            f"std_across_rounds={stacked.std(unbiased=False).item():.2f}"
        )


if __name__ == "__main__":
    main()
