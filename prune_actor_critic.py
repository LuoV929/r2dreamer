# prune_actor_critic.py
# Taylor 一阶重要性（合并 Actor / Value 第一层隐藏神经元），结构化剪枝 + 同步剪枝
# frozen / slow 副本，保存剪枝模型。路径请按 Colab / 本机修改。
#
# 依赖: torch_pruning, hydra, omegaconf；与 prune_reward 流程一致。

from __future__ import annotations

import copy
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.nn as nn
import torch_pruning as tp
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from pruning_utils import RMSNormPruningHandler

# ================================================================
# 路径与 Hydra（Colab 改这里；本机若目录不存在会在 __main__ 回落到 _REPO）
# ================================================================
CHECKPOINT_PATH = pathlib.Path(
    "/content/drive/MyDrive/r2dreamer_checkpoints/size100m_atari_breakout/latest.pt"
)
SAVE_DIR = pathlib.Path("/content/drive/MyDrive/r2dreamer_checkpoints/pruned_models_actor_critic")
CONFIG_DIR = "/content/r2dreamer/configs"

HYDRA_OVERRIDES = [
    "env=atari",
    "model=size100M",
    "model.rep_loss=dreamer",
    "model.compile=False",
    "device=cuda:0",
    "buffer.storage_device=cpu",
]

# 重要性估计：单独 (B,T)，避免 _cal_grad 反传 OOM。仍用完整损失图，仅缩小 slice。
# 若仍 OOM，改为 (2, 8) 或 (4, 8)，或减小 CALIB_N_BATCHES。
CALIB_BATCH_SIZE = 4
CALIB_BATCH_LENGTH = 16
CALIB_N_BATCHES = 16


def _build_cfg():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="configs", overrides=HYDRA_OVERRIDES)


def collect_data_and_compute_importance(agent: Dreamer, cfg, train_envs):
    """小 buffer 采数据；用缩小的 (B,T) 多次 _cal_grad 累积梯度，降低单次反传显存。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bs, bl = CALIB_BATCH_SIZE, CALIB_BATCH_LENGTH
    small_buffer_cfg = OmegaConf.create({
        "batch_size": bs,
        "batch_length": bl,
        "max_size": max(2000, bs * bl * 4),
        "device": str(cfg.device),
        "storage_device": "cpu",
    })
    replay_buffer = Buffer(small_buffer_cfg)

    min_steps = bs * bl + 1
    agent_state = agent.get_initial_state(cfg.env.env_num)
    act = agent_state["prev_action"].clone()
    steps_collected = 0
    done = torch.ones(cfg.env.env_num, dtype=torch.bool, device=cfg.device)

    print("  收集校准数据...")
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

    print(f"  已收集 {steps_collected} 步数据")

    agent.train()
    agent.zero_grad(set_to_none=True)

    print(
        f"  累积 {CALIB_N_BATCHES} 个 batch 的梯度 "
        f"(B={bs}, T={bl}，小于训练 batch 以降低显存)..."
    )
    for i in range(CALIB_N_BATCHES):
        data, index, initial = replay_buffer.sample()
        data = data.to(cfg.device)
        initial = (initial[0].to(cfg.device), initial[1].to(cfg.device))
        data = agent.preprocess(data)
        agent._cal_grad(data, initial)
        if torch.cuda.is_available() and (i + 1) % 4 == 0:
            torch.cuda.empty_cache()

    la = agent.actor.mlp.layers.actor_linear0
    lv = agent.value.mlp.layers.value_linear0
    if la.weight.shape != lv.weight.shape:
        raise RuntimeError(f"Actor/Value 第一层形状不一致: {la.weight.shape} vs {lv.weight.shape}")

    importance = (
        torch.abs(la.weight.grad * la.weight).sum(dim=1)
        + torch.abs(lv.weight.grad * lv.weight).sum(dim=1)
    ).detach().cpu()

    imp_min = importance.min()
    imp_max = importance.max()
    importance = (importance - imp_min) / (imp_max - imp_min + 1e-8)
    print(f"  合并重要性完成，均值={importance.mean():.4f}")
    return importance


def _group_index_for_linear(dg: tp.DependencyGraph, name_substr: str) -> int:
    """在 ``get_all_groups()`` 中找到 ``str(group)`` 含 ``name_substr`` 的组索引。

    多层 MLP 中 ``groups[1]`` 往往对应靠近输出的通道，不能写死。对 ``actor_linear{i}``、
    ``value_linear{i}`` 分别建图后匹配；多命中时取 ``hits[-1]``（与此前 linear0 经验一致）。
    """
    groups = list(dg.get_all_groups())
    hits = [i for i, g in enumerate(groups) if name_substr in str(g)]
    if not hits:
        preview = "\n".join(f"  [{i}] {str(g)[:240]}" for i, g in enumerate(groups))
        raise RuntimeError(f"依赖图中找不到含 {name_substr!r} 的组。当前组:\n{preview}")
    return hits[-1]


def _count_mlp_linears(mlp: nn.Module) -> int:
    return sum(1 for m in mlp.layers if isinstance(m, nn.Linear))


def _assert_uniform_mlp_linears(mlp: nn.Module, tag: str) -> int:
    """Dreamer 的 MLP 各隐藏层共用同一 ``units``；剪枝后所有 Linear 的 out_features 须一致。"""
    linears = [m for m in mlp.layers if isinstance(m, nn.Linear)]
    outs = [int(L.weight.shape[0]) for L in linears]
    if len(set(outs)) != 1:
        raise RuntimeError(
            f"{tag}: 剪枝后各 Linear 输出维不一致 {outs}，无法用单一 cfg.actor/critic.units 加载。"
            "应对 actor_linear0..2（及 value）逐层用同一 pruning_idxs 剪输出通道组。"
        )
    return outs[0]


def _prune_mlp_head_pair(
    live_mlp_last: nn.Sequential,
    frozen_mlp_last: nn.Sequential,
    dummy_feat: torch.Tensor,
    pruning_idxs: list[int],
    label: str,
    group_name_substr: str,
):
    """在含 ``group_name_substr``（如 ``actor_linear1``）的依赖组上剪枝；live / frozen 各剪一次。"""
    live_rg = [p.requires_grad for p in live_mlp_last.parameters()]
    frozen_rg = [p.requires_grad for p in frozen_mlp_last.parameters()]
    for p in live_mlp_last.parameters():
        p.requires_grad_(True)
    for p in frozen_mlp_last.parameters():
        p.requires_grad_(True)
    try:
        dg = tp.DependencyGraph()
        dg.build_dependency(
            live_mlp_last,
            example_inputs=dummy_feat,
            customized_pruners={nn.RMSNorm: RMSNormPruningHandler()},
        )
        groups = list(dg.get_all_groups())
        if len(groups) < 2:
            raise RuntimeError(f"{label}: 依赖组不足 2 个，当前 len={len(groups)}")
        gi = _group_index_for_linear(dg, group_name_substr)
        print(f"    [{label}] 依赖组 index={gi} (匹配 {group_name_substr!r})")
        groups[gi].prune(idxs=pruning_idxs)

        dg_f = tp.DependencyGraph()
        dg_f.build_dependency(
            frozen_mlp_last,
            example_inputs=dummy_feat,
            customized_pruners={nn.RMSNorm: RMSNormPruningHandler()},
        )
        groups_f = list(dg_f.get_all_groups())
        if len(groups_f) < 2:
            raise RuntimeError(f"{label} (frozen): 依赖组不足 2 个")
        gi_f = _group_index_for_linear(dg_f, group_name_substr)
        groups_f[gi_f].prune(idxs=pruning_idxs)
    finally:
        for p, r in zip(live_mlp_last.parameters(), live_rg):
            p.requires_grad_(r)
        for p, r in zip(frozen_mlp_last.parameters(), frozen_rg):
            p.requires_grad_(r)


def _prune_all_heads_same_idxs(agent: Dreamer, dummy_feat: torch.Tensor, pruning_idxs: list[int]) -> None:
    """对 Actor / Value / _slow 的每一层隐藏 Linear，用同一 ``pruning_idxs`` 剪输出通道（与单一 ``units`` 一致）。"""
    n_a = _count_mlp_linears(agent.actor.mlp)
    n_v = _count_mlp_linears(agent.value.mlp)
    n_s = _count_mlp_linears(agent._slow_value.mlp)
    if not (n_a == n_v == n_s):
        raise RuntimeError(f"Actor / Value / _slow_value 的 MLP 层数不一致: {n_a}, {n_v}, {n_s}")
    for li in range(n_a):
        _prune_mlp_head_pair(
            nn.Sequential(agent.actor.mlp, agent.actor.last),
            nn.Sequential(agent._frozen_actor.mlp, agent._frozen_actor.last),
            dummy_feat,
            pruning_idxs,
            f"actor_L{li}",
            f"actor_linear{li}",
        )
        _prune_mlp_head_pair(
            nn.Sequential(agent.value.mlp, agent.value.last),
            nn.Sequential(agent._frozen_value.mlp, agent._frozen_value.last),
            dummy_feat,
            pruning_idxs,
            f"value_L{li}",
            f"value_linear{li}",
        )
        _prune_mlp_head_pair(
            nn.Sequential(agent._slow_value.mlp, agent._slow_value.last),
            nn.Sequential(agent._frozen_slow_value.mlp, agent._frozen_slow_value.last),
            dummy_feat,
            pruning_idxs,
            f"_slow_L{li}",
            f"value_linear{li}",
        )


def prune_actor_critic(agent: Dreamer, importance: torch.Tensor, prune_ratio: float) -> Dreamer:
    """按合并重要性剪枝：Actor、Value、_slow_value 及其 frozen 镜像（第一层隐藏宽度一致）。"""
    n_neurons = len(importance)
    n_prune = int(n_neurons * prune_ratio)
    n_keep = n_neurons - n_prune
    _, sorted_indices = torch.sort(importance)
    pruning_idxs = sorted_indices[:n_prune].tolist()

    print(f"  总神经元: {n_neurons}, 剪掉: {n_prune}, 保留: {n_keep}")
    print(f"  被删神经元中最高重要性: {importance[sorted_indices[n_prune - 1]].item():.4f}")
    print(f"  被保留神经元中最低重要性: {importance[sorted_indices[n_prune]].item():.4f}")

    feat_size = agent.rssm.feat_size
    dummy_feat = torch.zeros(1, feat_size, device=agent.device)

    _prune_all_heads_same_idxs(agent, dummy_feat, pruning_idxs)

    ua = _assert_uniform_mlp_linears(agent.actor.mlp, "actor.mlp")
    uv = _assert_uniform_mlp_linears(agent.value.mlp, "value.mlp")
    us = _assert_uniform_mlp_linears(agent._slow_value.mlp, "_slow_value.mlp")
    if not (ua == uv == us):
        raise RuntimeError(f"Actor/Value/_slow_value 隐藏宽度不一致: {ua}, {uv}, {us}")
    print(f"  剪枝后各隐藏层统一 units = {ua} (actor / value / _slow_value 已对齐)")
    return agent


def verify_pruned_model(agent: Dreamer) -> bool:
    agent.eval()
    feat = torch.zeros(1, agent.rssm.feat_size, device=agent.device)
    try:
        with torch.no_grad():
            _ = agent.actor(feat)
            _ = agent.value(feat)
            _ = agent._frozen_actor(feat)
            _ = agent._frozen_value(feat)
            _ = agent._slow_value(feat)
            _ = agent._frozen_slow_value(feat)
        print("  Actor / Value / slow / frozen 前向均成功")
        return True
    except Exception as e:
        print(f"  验证失败: {e}")
        return False


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _build_cfg()
    cfg_backup = copy.deepcopy(cfg)
    train_envs, eval_envs, obs_space, act_space = make_envs(cfg.env)

    prune_ratios = [0.3, 0.5, 0.7]

    for prune_ratio in prune_ratios:
        print(f"\n{'=' * 60}\n剪枝率: {prune_ratio * 100:.0f}%\n{'=' * 60}")

        cfg_model = copy.deepcopy(cfg_backup.model)
        print("加载原始模型...")
        agent = Dreamer(cfg_model, obs_space, act_space).to(cfg.device)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=cfg.device)
        agent.load_state_dict(ckpt["agent_state_dict"])

        print("计算重要性...")
        importance = collect_data_and_compute_importance(agent, cfg, train_envs)

        print("执行剪枝...")
        agent = prune_actor_critic(agent, importance, prune_ratio)

        print("验证...")
        if not verify_pruned_model(agent):
            print("  跳过保存")
            continue

        agent.float()
        save_path = SAVE_DIR / f"actor_critic_pruned_{int(prune_ratio * 100)}pct.pt"
        n0 = agent.actor.mlp.layers.actor_linear0.weight.shape[0]
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "prune_ratio": prune_ratio,
                "pruned_modules": ["actor", "value", "_slow_value", "frozen mirrors"],
                "original_neurons": int(importance.numel()),
                "remaining_neurons": int(n0),
            },
            save_path,
        )
        print(f"  已保存: {save_path}")

    print("\n全部完成。")


if __name__ == "__main__":
    if not pathlib.Path(CONFIG_DIR).exists():
        CONFIG_DIR = str(_REPO / "configs")
        CHECKPOINT_PATH = _REPO / "latest.pt"
        SAVE_DIR = _REPO / "pruned_models_actor_critic"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    main()
