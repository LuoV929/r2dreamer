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
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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
def evaluate_once(agent, eval_envs) -> dict[str, float]:
    """One eval round: each parallel env runs until its first episode ends."""
    envs = eval_envs
    agent.eval()
    done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
    once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
    steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
    returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)

    agent_state = agent.get_initial_state(envs.env_num)
    act = agent_state["prev_action"].clone()

    while not once_done.all():
        steps += ~done * ~once_done
        act_cpu = act.detach().to("cpu")
        done_cpu = done.detach().to("cpu")
        trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
        trans = trans_cpu.to(agent.device, non_blocking=True)
        done = done_cpu.to(agent.device)

        trans["action"] = act
        act, agent_state = agent.act(trans, agent_state, eval=True)
        returns += trans["reward"][:, 0] * ~once_done
        once_done |= done

    agent.train()
    rets = returns.detach().cpu()
    return {
        "mean_return": float(rets.mean()),
        "std_return": float(rets.std(unbiased=False)),
        "min_return": float(rets.min()),
        "max_return": float(rets.max()),
        "mean_length": float(steps.to(torch.float32).mean()),
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
            m = evaluate_once(agent, eval_envs)
            all_means.append(m["mean_return"])
            print(
                f"  round {r + 1}/{args.num_rounds}: "
                f"mean_return={m['mean_return']:.2f} "
                f"(std_episode={m['std_return']:.2f}, len={m['mean_length']:.1f})"
            )

        stacked = torch.tensor(all_means)
        print(
            f"  aggregate over rounds: mean_of_round_means={stacked.mean().item():.2f}, "
            f"std_across_rounds={stacked.std(unbiased=False).item():.2f}"
        )


if __name__ == "__main__":
    main()
