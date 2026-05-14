# finetune_pruned_rssm_decoder.py
# 加载 ``prune_rssm_decoder.py`` 导出的 checkpoint，在线微调世界模型（RSSM + ConvDecoder）。
#
# 要点（与 ``finetune_pruned_actor_critic.py`` 一致）:
# - 关闭 Dreamer 内 autocast，避免与剪枝后模块的混精度问题。
# - Buffer 的 ``device`` 与 agent 一致（cuda），避免 ``update`` 内 CPU/GPU 混用。
# - 仅把 ``model.hidden`` / ``model.rssm.hidden`` 与 ``model.decoder.cnn.units`` 改成剪枝后宽度；
#   **不要**改 ``model.units``（Actor / Value / Reward / Encoder MLP 仍为原 768 等）。

from __future__ import annotations

import contextlib
import copy
import pathlib
import sys

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dreamer as dreamer_module

dreamer_module.autocast = lambda *args, **kwargs: contextlib.nullcontext()

from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
import tools

# ================================================================
# 路径（Colab 请改成你的 Drive 路径）
# ================================================================
PRUNED_MODELS_DIR = pathlib.Path(
    "/content/drive/MyDrive/r2dreamer_checkpoints/pruned_models_rssm_decoder"
)
FINETUNED_DIR = pathlib.Path(
    "/content/drive/MyDrive/r2dreamer_checkpoints/finetuned_models_rssm_decoder"
)
CONFIG_DIR = "/content/r2dreamer/configs"

FINETUNED_DIR.mkdir(parents=True, exist_ok=True)

HYDRA_OVERRIDES = [
    "env=atari",
    "model=size100M",
    "model.rep_loss=dreamer",
    "model.compile=False",
    "device=cuda:0",
    "buffer.storage_device=cpu",
    "trainer.steps=100000",
    "trainer.eval_episode_num=0",
]


def _compose_cfg():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="configs", overrides=HYDRA_OVERRIDES)


def _infer_hidden_width(state_dict: dict, ckpt: dict) -> int:
    if "rssm_hidden_after" in ckpt:
        return int(ckpt["rssm_hidden_after"])
    k_obs = "rssm._obs_net.obs_net_0.weight"
    if k_obs not in state_dict:
        raise KeyError(f"无法推断 hidden：缺少 {k_obs}")
    return int(state_dict[k_obs].shape[0])


def load_pruned_agent(pruned_ckpt_path: pathlib.Path, cfg_backup, obs_space, act_space) -> Dreamer:
    ckpt = torch.load(pruned_ckpt_path, map_location="cpu")
    state_dict = ckpt["agent_state_dict"]

    for k in list(state_dict.keys()):
        t = state_dict[k]
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            state_dict[k] = t.float()

    H = _infer_hidden_width(state_dict, ckpt)

    k_dyn = "rssm._deter_net._dyn_in0.0.weight"
    k_fdyn = "_frozen_rssm._deter_net._dyn_in0.0.weight"
    k_sp1 = "decoder._cnn.sp1.0.weight"
    if k_dyn not in state_dict or k_sp1 not in state_dict:
        raise KeyError(f"期望 Dreamer 剪枝 ckpt 含 {k_dyn} 与 {k_sp1}")
    if int(state_dict[k_dyn].shape[0]) != H:
        raise RuntimeError(f"dyn_in0 输出维 {state_dict[k_dyn].shape[0]} 与推断 H={H} 不一致")
    if int(state_dict[k_sp1].shape[0]) != 2 * H:
        raise RuntimeError(
            f"decoder._cnn.sp1[0] 输出维 {state_dict[k_sp1].shape[0]} 与 2*H={2 * H} 不一致"
        )
    if k_fdyn in state_dict and int(state_dict[k_fdyn].shape[0]) != H:
        raise RuntimeError("_frozen_rssm 与 rssm 的 dyn_in0 宽度不一致")

    print(f"  检测到 RSSM/Decoder CNN hidden 宽度: H={H}（Actor 等仍用 cfg.model.units，不修改）")

    cfg_model = copy.deepcopy(cfg_backup.model)
    cfg_model.hidden = H
    if hasattr(cfg_model, "rssm") and hasattr(cfg_model.rssm, "hidden"):
        cfg_model.rssm.hidden = H
    if hasattr(cfg_model, "decoder") and hasattr(cfg_model.decoder, "cnn"):
        cfg_model.decoder.cnn.units = H

    agent = Dreamer(cfg_model, obs_space, act_space).to(cfg_backup.device)
    missing, unexpected = agent.load_state_dict(state_dict, strict=True)
    agent.float()
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict: missing={missing}, unexpected={unexpected}")
    print("  模型加载成功")
    return agent


def main():
    cfg = _compose_cfg()
    if cfg.model.rep_loss != "dreamer":
        raise RuntimeError("请在 HYDRA_OVERRIDES 中设置 model.rep_loss=dreamer")
    cfg_backup = copy.deepcopy(cfg)
    train_envs, eval_envs, obs_space, act_space = make_envs(cfg.env)

    # 与 prune_rssm_decoder 保存名 ``rssm_decoder_pruned_{pct}pct.pt`` 对齐（pct = int(ratio*100)）
    # prune_pcts = [30, 50, 70]
    # prune_pcts = [55, 60, 65, 75]
    prune_pcts = [60, 65]

    for pct in prune_pcts:
        print(f"\n{'=' * 60}\n微调 RSSM+Decoder 剪枝率 {pct}%\n{'=' * 60}")
        pruned_path = PRUNED_MODELS_DIR / f"rssm_decoder_pruned_{pct}pct.pt"
        finetune_logdir = FINETUNED_DIR / f"rssm_decoder_{pct}pct"
        finetune_logdir.mkdir(parents=True, exist_ok=True)

        if not pruned_path.is_file():
            print(f"  跳过：找不到 {pruned_path}")
            continue

        try:
            agent = load_pruned_agent(pruned_path, cfg_backup, obs_space, act_space)
        except Exception as e:
            print(f"  加载失败: {e}")
            continue

        print(
            f"  rssm._hidden={agent.rssm._hidden}, "
            f"decoder._cnn.units={agent.decoder._cnn.units}, "
            f"sp1[0].out={agent.decoder._cnn.sp1[0].out_features}"
        )

        small_buffer_cfg = OmegaConf.create({
            "batch_size": cfg.batch_size,
            "batch_length": cfg.batch_length,
            "max_size": 2000,
            "device": str(cfg.device),
            "storage_device": "cpu",
        })
        replay_buffer = Buffer(small_buffer_cfg)

        min_steps = cfg.batch_size * cfg.batch_length + 1
        agent_state = agent.get_initial_state(cfg.env.env_num)
        act = agent_state["prev_action"].clone()
        steps_collected = 0
        done = torch.ones(cfg.env.env_num, dtype=torch.bool, device=cfg.device)

        print("预填充 replay buffer...")
        while steps_collected < min_steps:
            act_cpu = act.detach().to("cpu")
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = train_envs.step(act_cpu, done_cpu)
            trans = trans_cpu.to(cfg.device, non_blocking=True)
            done = done_cpu.to(cfg.device)
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = torch.zeros(cfg.env.env_num, dtype=torch.int32, device=cfg.device)
            replay_buffer.add_transition(trans.detach().cpu())
            steps_collected += int((~done).sum())

        logger = tools.Logger(finetune_logdir)
        agent.train()
        step = 0
        done = torch.ones(cfg.env.env_num, dtype=torch.bool, device=cfg.device)
        returns = torch.zeros(cfg.env.env_num, dtype=torch.float32, device=cfg.device)
        lengths = torch.zeros(cfg.env.env_num, dtype=torch.int32, device=cfg.device)
        episode_ids = torch.arange(cfg.env.env_num, dtype=torch.int32, device=cfg.device)
        agent_state = agent.get_initial_state(cfg.env.env_num)
        act = agent_state["prev_action"].clone()

        from tools import Every

        updates_needed = Every(
            cfg.batch_size * cfg.batch_length / cfg.trainer.train_ratio * cfg.env.action_repeat
        )
        should_save = Every(50000)

        finetune_steps = int(cfg_backup.trainer.steps)

        while step < finetune_steps:
            act_cpu = act.detach().to("cpu")
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = train_envs.step(act_cpu, done_cpu)
            trans = trans_cpu.to(cfg.device, non_blocking=True)
            done = done_cpu.to(cfg.device)
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids
            replay_buffer.add_transition(trans.detach().cpu())
            returns += trans["reward"][:, 0]
            step += int((~done).sum()) * cfg.env.action_repeat
            lengths += ~done

            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        print(f"  [step {step}] episode/score={returns[i].item():.1f} / len={lengths[i].item()}")
                        returns[i] = 0
                        lengths[i] = 0

            if step // cfg.env.action_repeat > cfg.batch_length + 1:
                for _ in range(updates_needed(step)):
                    agent.update(replay_buffer)

            if should_save(step):
                torch.save({"agent_state_dict": agent.state_dict()}, finetune_logdir / f"checkpoint_step{step}.pt")
                print(f"  [step {step}] checkpoint 已保存")

        final_path = FINETUNED_DIR / f"rssm_decoder_{pct}pct_finetuned.pt"
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "prune_ratio": pct / 100.0,
                "pruned_modules": ["rssm", "decoder._cnn", "_frozen_rssm"],
                "rssm_hidden_after": int(agent.rssm._hidden),
                "finetune_steps": finetune_steps,
            },
            final_path,
        )
        print(f"微调完成: {final_path}")

    print("\n全部微调完成。")


if __name__ == "__main__":
    if not pathlib.Path(CONFIG_DIR).exists():
        CONFIG_DIR = str(_ROOT / "configs")
        PRUNED_MODELS_DIR = _ROOT / "pruned_models_rssm_decoder"
        FINETUNED_DIR = _ROOT / "finetuned_models_rssm_decoder"
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    main()
