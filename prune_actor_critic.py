# prune_actor_critic.py
# Taylor 一阶重要性（合并 Actor / Value 的 linear0），按同一组神经元索引做
# 「结构化宽度剪枝」：各隐藏层与 RMSNorm、last 同步缩维，与 Hydra 单一 units 一致。
#
# 依赖: hydra, omegaconf（不再用 torch_pruning 做多层剪枝，避免依赖组索引不稳定）。

from __future__ import annotations

import copy
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs

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

CALIB_BATCH_SIZE = 4
CALIB_BATCH_LENGTH = 16
CALIB_N_BATCHES = 16


def _build_cfg():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="configs", overrides=HYDRA_OVERRIDES)


def collect_data_and_compute_importance(agent: Dreamer, cfg, train_envs):
    """小 buffer；缩小 (B,T) 多次 _cal_grad 累积梯度。"""
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


def _list_mlp_linears_in_order(mlp: nn.Module) -> list[tuple[str, nn.Linear]]:
    pairs = [(n, m) for n, m in mlp.layers.named_children() if isinstance(m, nn.Linear)]
    pairs.sort(key=lambda nm: int(nm[0].split("linear")[-1]))
    return pairs


def _manual_neuron_prune_mlp_and_last(
    mlp: nn.Module,
    last: nn.Linear,
    prune_idxs: list[int],
) -> None:
    """按神经元索引 ``prune_idxs`` 从各隐藏层同步删掉通道，保持各层 out 维一致。

    约定：各 ``*_linear*`` 的 out_features 原为同一 ``units``；``prune_idxs`` 为要删的
    输出通道下标（与 Taylor 在 linear0 上的打分一致）。
    """
    rm = {int(i) for i in prune_idxs}
    linears = _list_mlp_linears_in_order(mlp)
    if not linears:
        raise RuntimeError("MLP 中未找到 Linear")

    d0 = linears[0][1].out_features
    keep_list = [i for i in range(d0) if i not in rm]
    if not keep_list:
        raise RuntimeError("剪枝后无剩余通道")
    dev = linears[0][1].weight.device
    keep = torch.tensor(keep_list, dtype=torch.long, device=dev)

    for li, (n, lin) in enumerate(linears):
        W = lin.weight.data
        b = lin.bias.data if lin.bias is not None else None
        if li == 0:
            Wn = W.index_select(0, keep)
            bn = b.index_select(0, keep) if b is not None else None
        else:
            Wn = W.index_select(1, keep).index_select(0, keep)
            bn = b.index_select(0, keep) if b is not None else None
        lin.weight = nn.Parameter(Wn.contiguous().clone())
        if bn is not None:
            lin.bias = nn.Parameter(bn.contiguous().clone())
        lin.out_features = Wn.shape[0]
        lin.in_features = Wn.shape[1]

        norm_name = n.replace("linear", "norm")
        norm = getattr(mlp.layers, norm_name)
        if isinstance(norm, nn.RMSNorm):
            nw = norm.weight.data.index_select(0, keep)
            norm.weight = nn.Parameter(nw.contiguous().clone())
            norm.normalized_shape = (len(keep_list),)

    Wl = last.weight.data
    Wln = Wl.index_select(1, keep)
    last.weight = nn.Parameter(Wln.contiguous().clone())
    last.in_features = Wln.shape[1]
    last.out_features = Wln.shape[0]


def _manual_prune_mlphead(head: nn.Module, prune_idxs: list[int]) -> None:
    """``head`` 为 ``MLPHead``：含 ``.mlp`` 与 ``.last``。"""
    _manual_neuron_prune_mlp_and_last(head.mlp, head.last, prune_idxs)


def _assert_uniform_mlp_linears(mlp: nn.Module, tag: str) -> int:
    linears = [m for m in mlp.layers if isinstance(m, nn.Linear)]
    outs = [int(L.weight.shape[0]) for L in linears]
    if len(set(outs)) != 1:
        raise RuntimeError(f"{tag}: 各 Linear 输出维不一致 {outs}")
    return outs[0]


def prune_actor_critic(agent: Dreamer, importance: torch.Tensor, prune_ratio: float) -> Dreamer:
    """Actor / Value / _slow 及 frozen 镜像：同一 ``prune_idxs`` 手动同步剪枝。"""
    n_neurons = len(importance)
    n_prune = int(n_neurons * prune_ratio)
    n_keep = n_neurons - n_prune
    _, sorted_indices = torch.sort(importance)
    pruning_idxs = sorted_indices[:n_prune].tolist()

    print(f"  总神经元: {n_neurons}, 剪掉: {n_prune}, 保留: {n_keep}")
    print(f"  被删神经元中最高重要性: {importance[sorted_indices[n_prune - 1]].item():.4f}")
    print(f"  被保留神经元中最低重要性: {importance[sorted_indices[n_prune]].item():.4f}")
    print("  使用手动同构剪枝（index_select，不依赖 torch_pruning 依赖组）...")

    for h in (
        agent.actor,
        agent._frozen_actor,
        agent.value,
        agent._frozen_value,
        agent._slow_value,
        agent._frozen_slow_value,
    ):
        _manual_prune_mlphead(h, pruning_idxs)

    ua = _assert_uniform_mlp_linears(agent.actor.mlp, "actor.mlp")
    uv = _assert_uniform_mlp_linears(agent.value.mlp, "value.mlp")
    us = _assert_uniform_mlp_linears(agent._slow_value.mlp, "_slow_value.mlp")
    if not (ua == uv == us):
        raise RuntimeError(f"Actor/Value/_slow_value 隐藏宽度不一致: {ua}, {uv}, {us}")
    print(f"  剪枝后各隐藏层统一 units = {ua}")
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

    # prune_ratios = [0.3, 0.5, 0.7]
    prune_ratios = [0.55, 0.6, 0.65]

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
