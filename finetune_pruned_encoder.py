# finetune_pruned_encoder.py
# 加载 ``prune_encoder.py`` 导出的 checkpoint 并在线微调。
#
# 要点：与 ``finetune_pruned_rssm_decoder.py`` 相同——关闭 autocast、buffer device 与 agent 一致；
# 通过 ``encoder.cnn.last_out_channels``（见 ``networks.ConvEncoder``）重建与剪枝后一致的 CNN。

from __future__ import annotations

import contextlib
import copy
import pathlib
import re
import sys

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dreamer as dreamer_module

dreamer_module.autocast = lambda *args, **kwargs: contextlib.nullcontext()

from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
import tools

PRUNED_MODELS_DIR = pathlib.Path(
    "/content/drive/MyDrive/r2dreamer_checkpoints/pruned_models_encoder"
)
FINETUNED_DIR = pathlib.Path(
    "/content/drive/MyDrive/r2dreamer_checkpoints/finetuned_models_encoder"
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


def _infer_last_conv_out_channels(state_dict: dict) -> int | None:
    last_i = -1
    last_c: int | None = None
    for k, t in state_dict.items():
        if not isinstance(t, torch.Tensor) or t.dim() != 4:
            continue
        m = re.match(r"encoder\.encoders\.0\.layers\.(\d+)\.weight$", k)
        if not m:
            continue
        li = int(m.group(1))
        if li > last_i:
            last_i = li
            last_c = int(t.shape[0])
    return last_c


def load_pruned_agent(ckpt: dict, cfg_backup, obs_space, act_space) -> Dreamer:
    state_dict = ckpt["agent_state_dict"]

    for k in list(state_dict.keys()):
        t = state_dict[k]
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            state_dict[k] = t.float()

    C_meta = ckpt.get("encoder_cnn_last_out_channels_after")
    C_sd = _infer_last_conv_out_channels(state_dict)
    if C_meta is not None and C_sd is not None and int(C_meta) != int(C_sd):
        raise RuntimeError(f"ckpt 元数据 last_out_channels={C_meta} 与 state_dict 推断 {C_sd} 不一致")
    C_target = int(C_meta) if C_meta is not None else (int(C_sd) if C_sd is not None else None)
    if C_target is None:
        raise RuntimeError("无法从 checkpoint 推断 encoder 末层卷积输出通道数")

    cfg_model = copy.deepcopy(cfg_backup.model)
    depth = int(cfg_model.encoder.cnn.depth)
    mults = list(cfg_model.encoder.cnn.mults)
    c_def = depth * int(mults[-1])
    if C_target == c_def:
        print(f"  末层通道 {C_target} 与默认一致，无需 last_out_channels 覆盖")
    else:
        with open_dict(cfg_model.encoder.cnn):
            cfg_model.encoder.cnn.last_out_channels = C_target
        print(f"  设置 encoder.cnn.last_out_channels={C_target}（默认末层 {c_def}）")

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

    # prune_pcts = [30, 50, 70]
    prune_pcts = [80, 85, 90, 95]

    for pct in prune_pcts:
        print(f"\n{'=' * 60}\n微调 Encoder 剪枝率 {pct}%\n{'=' * 60}")
        pruned_path = PRUNED_MODELS_DIR / f"encoder_pruned_{pct}pct.pt"
        finetune_logdir = FINETUNED_DIR / f"encoder_{pct}pct"
        finetune_logdir.mkdir(parents=True, exist_ok=True)

        if not pruned_path.is_file():
            print(f"  跳过：找不到 {pruned_path}")
            continue

        ckpt_meta = torch.load(pruned_path, map_location="cpu")
        try:
            agent = load_pruned_agent(ckpt_meta, cfg_backup, obs_space, act_space)
        except Exception as e:
            print(f"  加载失败: {e}")
            continue

        cnn = agent.encoder.encoders[0]
        print(f"  embed_size={agent.embed_size}, ConvEncoder.out_dim={cnn.out_dim}")

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

        final_path = FINETUNED_DIR / f"encoder_{pct}pct_finetuned.pt"
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "prune_ratio": pct / 100.0,
                "pruned_module": "encoder_cnn_last_conv",
                "encoder_cnn_last_out_channels_after": ckpt_meta.get("encoder_cnn_last_out_channels_after"),
                "embed_dim_after": int(agent.embed_size),
                "finetune_steps": finetune_steps,
            },
            final_path,
        )
        print(f"微调完成: {final_path}")

    print("\n全部微调完成。")


if __name__ == "__main__":
    if not pathlib.Path(CONFIG_DIR).exists():
        CONFIG_DIR = str(_ROOT / "configs")
        PRUNED_MODELS_DIR = _ROOT / "pruned_models_encoder"
        FINETUNED_DIR = _ROOT / "finetuned_models_encoder"
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    main()
