# prune_encoder.py
# DreamerV3（rep_loss=dreamer）：对 **CNN Encoder 最后一层卷积的输出通道** 做结构化剪枝，并同步
# ``RSSM._obs_net`` 首层 Linear 在 **embedding 段** 的输入列（``concat(deter, embed)`` 中 embed 对应列）。
#
# 设计要点
# ---------
# 1) 仅支持 ``MultiEncoder`` 仅含 **单个 ConvEncoder**（Atari 等 ``mlp_keys`` 为空）。
# 2) 重要性：最后一层 ``Conv2dSamePad`` 的每个输出通道，Taylor ``sum(|grad*w|)`` 在 (C_in,k,k) 上聚合。
# 3) 剪枝后更新 ``ConvEncoder.out_dim``、``MultiEncoder.out_dim``、``Dreamer.embed_size``；
#    同步 ``_frozen_encoder`` 与 ``_frozen_rssm._obs_net``（与 ``act`` 中 frozen 路径一致）。
# 4) 重建 checkpoint 时需 ``encoder.cnn.last_out_channels``，见 ``networks.ConvEncoder`` 与 ``finetune_pruned_encoder.py``。
#
# Colab：改 ``CHECKPOINT_PATH`` / ``SAVE_DIR`` / ``CONFIG_DIR``；本机无 ``/content/...`` 时在 ``__main__`` 回落仓库路径。

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
from networks import Conv2dSamePad, ConvEncoder, RMSNorm2D

CHECKPOINT_PATH = pathlib.Path(
    "/content/drive/MyDrive/r2dreamer_checkpoints/size100m_atari_breakout/latest.pt"
)
SAVE_DIR = pathlib.Path("/content/drive/MyDrive/r2dreamer_checkpoints/pruned_models_encoder")
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


def _get_cnn_encoder_from_multi(me: nn.Module) -> ConvEncoder:
    enc = me
    if len(enc.encoders) != 1:
        raise RuntimeError(f"本脚本仅支持单路 Encoder，当前 encoders 数量={len(enc.encoders)}")
    c0 = enc.encoders[0]
    if not isinstance(c0, ConvEncoder):
        raise RuntimeError(f"首路 Encoder 应为 ConvEncoder，实为 {type(c0)}")
    return c0


def _get_cnn_encoder(agent: Dreamer) -> ConvEncoder:
    return _get_cnn_encoder_from_multi(agent.encoder)


def _last_conv_and_norm_index(enc: ConvEncoder) -> tuple[Conv2dSamePad, int, RMSNorm2D | None, int]:
    last_conv: Conv2dSamePad | None = None
    last_i = -1
    for i, m in enumerate(enc.layers):
        if isinstance(m, Conv2dSamePad):
            last_conv, last_i = m, i
    if last_conv is None:
        raise RuntimeError("ConvEncoder 中未找到 Conv2dSamePad")
    norm_mod: RMSNorm2D | None = None
    norm_i = -1
    for j in range(last_i + 1, min(last_i + 5, len(enc.layers))):
        if isinstance(enc.layers[j], RMSNorm2D):
            norm_mod, norm_i = enc.layers[j], j
            break
    return last_conv, last_i, norm_mod, norm_i


def collect_data_and_compute_importance(agent: Dreamer, cfg, train_envs) -> torch.Tensor:
    if agent.rep_loss != "dreamer":
        raise RuntimeError("本脚本仅支持 rep_loss=dreamer")
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

    print(f"  累积 {CALIB_N_BATCHES} 个 batch 的梯度 (B={bs}, T={bl})...")
    for i in range(CALIB_N_BATCHES):
        data, index, initial = replay_buffer.sample()
        data = data.to(cfg.device)
        initial = (initial[0].to(cfg.device), initial[1].to(cfg.device))
        data = agent.preprocess(data)
        agent._cal_grad(data, initial)
        if torch.cuda.is_available() and (i + 1) % 4 == 0:
            torch.cuda.empty_cache()

    cnn = _get_cnn_encoder(agent)
    last_conv, _, _, _ = _last_conv_and_norm_index(cnn)
    g = last_conv.weight.grad
    if g is None:
        imp = torch.zeros(last_conv.out_channels, device=last_conv.weight.device)
    else:
        imp = torch.abs(g * last_conv.weight).flatten(1).sum(dim=1).detach()
    imp = imp.float().cpu()
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
    print(f"  Encoder 末层卷积通道重要性完成，通道数={imp.numel()}，均值={imp.mean():.4f}")
    return imp


def _keep_channel_indices(C: int, prune_idxs: list[int]) -> list[int]:
    rm = {int(i) for i in prune_idxs}
    keep = [i for i in range(C) if i not in rm]
    if not keep:
        raise RuntimeError("剪枝后须至少保留 1 个输出通道")
    return keep


def _flat_embed_indices_for_removed_channels(S: int, C: int, rm_ch: set[int]) -> set[int]:
    out: set[int] = set()
    for c in rm_ch:
        if 0 <= c < C:
            base = c * S
            out.update(range(base, base + S))
    return out


def _prune_conv2d_out(conv: Conv2dSamePad, keep: torch.Tensor) -> None:
    W = conv.weight.data.index_select(0, keep)
    b = conv.bias.data.index_select(0, keep)
    conv.weight = nn.Parameter(W.contiguous().clone())
    conv.bias = nn.Parameter(b.contiguous().clone())
    conv.out_channels = int(W.shape[0])


def _prune_rmsnorm2d(norm: RMSNorm2D, keep: torch.Tensor) -> None:
    nw = norm.weight.data.index_select(0, keep)
    norm.weight = nn.Parameter(nw.contiguous().clone())
    norm.normalized_shape = (int(keep.numel()),)


def _prune_obs_net_embed_columns(
    obs_net: nn.Sequential,
    deter: int,
    E_old: int,
    rm_ch: set[int],
    C_old: int,
    dev: torch.device,
) -> int:
    S = E_old // C_old
    if S * C_old != E_old:
        raise RuntimeError(f"embed_dim {E_old} 无法整除末层通道 {C_old}")
    rm_flat = _flat_embed_indices_for_removed_channels(S, C_old, rm_ch)
    keep_emb = [i for i in range(E_old) if i not in rm_flat]
    keep_emb_t = torch.tensor(keep_emb, dtype=torch.long, device=dev)
    cols = torch.cat([torch.arange(deter, device=dev, dtype=torch.long), deter + keep_emb_t])
    lin = getattr(obs_net, "obs_net_0")
    if not isinstance(lin, nn.Linear):
        raise RuntimeError("期望 obs_net_0 为 Linear")
    W = lin.weight.data.index_select(1, cols)
    b = lin.bias.data
    lin.weight = nn.Parameter(W.contiguous().clone())
    lin.in_features = int(W.shape[1])
    E_new = len(keep_emb)
    return E_new


def _apply_encoder_prune_to_agent(agent: Dreamer, prune_idxs: list[int]) -> tuple[int, int, int]:
    """返回 (C_old, C_new, E_new)。"""
    cnn = _get_cnn_encoder(agent)
    last_conv, _, norm, _ = _last_conv_and_norm_index(cnn)
    C_old = last_conv.out_channels
    E_old = int(agent.encoder.out_dim)
    S = E_old // C_old
    if S * C_old != E_old:
        raise RuntimeError(f"out_dim={E_old} 与末层通道 {C_old} 不一致，无法推断空间乘子")

    rm_ch = {int(i) for i in prune_idxs}
    keep_list = _keep_channel_indices(C_old, prune_idxs)
    dev = last_conv.weight.device
    keep = torch.tensor(keep_list, dtype=torch.long, device=dev)
    C_new = len(keep_list)

    _prune_conv2d_out(last_conv, keep)
    if norm is not None:
        _prune_rmsnorm2d(norm, keep)

    cnn.out_dim = C_new * S
    agent.encoder.out_dim = cnn.out_dim
    deter = int(agent.rssm._deter)

    E_new = _prune_obs_net_embed_columns(agent.rssm._obs_net, deter, E_old, rm_ch, C_old, dev)
    _prune_obs_net_embed_columns(agent._frozen_rssm._obs_net, deter, E_old, rm_ch, C_old, dev)

    # 同步 CNN 权重到 _frozen_encoder
    fcnn = _get_cnn_encoder_from_multi(agent._frozen_encoder)
    flast, _, fnorm, _ = _last_conv_and_norm_index(fcnn)
    _prune_conv2d_out(flast, keep)
    if fnorm is not None and norm is not None:
        _prune_rmsnorm2d(fnorm, keep)
    fcnn.out_dim = C_new * S
    agent._frozen_encoder.out_dim = agent.encoder.out_dim

    agent.embed_size = int(agent.encoder.out_dim)
    return C_old, C_new, E_new


def prune_encoder(agent: Dreamer, importance: torch.Tensor, prune_ratio: float) -> Dreamer:
    C = int(importance.numel())
    n_prune = int(C * prune_ratio)
    n_keep = C - n_prune
    _, sorted_indices = torch.sort(importance)
    pruning_idxs = sorted_indices[:n_prune].tolist()

    print(f"  末层卷积输出通道: {C}, 剪掉: {n_prune}, 保留: {n_keep}")
    if n_prune > 0:
        print(f"  被删通道中最高重要性: {importance[sorted_indices[n_prune - 1]].item():.4f}")
        print(f"  被保留通道中最低重要性: {importance[sorted_indices[n_prune]].item():.4f}")

    C_old, C_new, E_new = _apply_encoder_prune_to_agent(agent, pruning_idxs)
    print(f"  剪枝后末层通道 {C_old} -> {C_new}，embed_dim -> {E_new}，embed_size={agent.embed_size}")
    return agent


def verify_pruned_model(agent: Dreamer, obs_space, act_dim: int) -> bool:
    agent.eval()
    if not hasattr(obs_space, "spaces") or "image" not in obs_space.spaces:
        print("  验证跳过：obs_space 无 Dict+image")
        return False
    im_shape = tuple(obs_space.spaces["image"].shape)
    B, T = 2, 5
    try:
        data = {
            "image": torch.zeros(B, T, *im_shape, device=agent.device, dtype=torch.float32),
            "action": torch.zeros(B, T, act_dim, device=agent.device, dtype=torch.float32),
            "is_first": torch.zeros(B, T, dtype=torch.bool, device=agent.device),
        }
        data = agent.preprocess(data)
        embed = agent.encoder(data)
        assert embed.shape[-1] == agent.embed_size
        initial = agent.rssm.initial(B)
        post_stoch, post_deter, _ = agent.rssm.observe(
            embed, data["action"], initial, data["is_first"]
        )
        if agent.rep_loss == "dreamer":
            _ = agent.decoder(post_stoch, post_deter)
        fe = agent._frozen_encoder(data)
        assert fe.shape[-1] == agent.embed_size
        print("  Encoder + RSSM (+Decoder) 前向成功")
        return True
    except Exception as e:
        print(f"  验证失败: {e}")
        return False


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _build_cfg()
    if cfg.model.rep_loss != "dreamer":
        raise RuntimeError("请在 HYDRA_OVERRIDES 中设置 model.rep_loss=dreamer")

    cfg_backup = copy.deepcopy(cfg)
    train_envs, eval_envs, obs_space, act_space = make_envs(cfg.env)
    act_dim = act_space.n if hasattr(act_space, "n") else int(sum(act_space.shape))

    # prune_ratios = [0.3, 0.5, 0.7]
    prune_ratios = [0.8, 0.85, 0.9, 0.95]

    for prune_ratio in prune_ratios:
        print(f"\n{'=' * 60}\nEncoder 通道剪枝率: {prune_ratio * 100:.0f}%\n{'=' * 60}")

        cfg_model = copy.deepcopy(cfg_backup.model)
        print("加载原始模型...")
        agent = Dreamer(cfg_model, obs_space, act_space).to(cfg.device)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=cfg.device)
        agent.load_state_dict(ckpt["agent_state_dict"])

        print("计算重要性...")
        importance = collect_data_and_compute_importance(agent, cfg, train_envs)
        cnn0 = _get_cnn_encoder(agent)
        last_conv, _, _, _ = _last_conv_and_norm_index(cnn0)
        C0 = int(last_conv.out_channels)

        print("执行剪枝...")
        agent = prune_encoder(agent, importance, prune_ratio)

        print("验证...")
        if not verify_pruned_model(agent, obs_space, act_dim):
            print("  跳过保存")
            continue

        agent.float()
        last_conv_f, _, _, _ = _last_conv_and_norm_index(_get_cnn_encoder(agent))
        C1 = int(last_conv_f.out_channels)
        save_path = SAVE_DIR / f"encoder_pruned_{int(prune_ratio * 100)}pct.pt"
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "prune_ratio": prune_ratio,
                "pruned_module": "encoder_cnn_last_conv",
                "encoder_cnn_last_out_channels_before": C0,
                "encoder_cnn_last_out_channels_after": C1,
                "embed_dim_after": int(agent.embed_size),
            },
            save_path,
        )
        print(f"  已保存: {save_path}")

    print("\n全部完成。")


if __name__ == "__main__":
    if not pathlib.Path(CONFIG_DIR).exists():
        CONFIG_DIR = str(_REPO / "configs")
        CHECKPOINT_PATH = _REPO / "latest.pt"
        SAVE_DIR = _REPO / "pruned_models_encoder"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    main()
