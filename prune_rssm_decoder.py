# prune_rssm_decoder.py
# DreamerV3（rep_loss=dreamer）：对 RSSM 与 ConvDecoder 的「共享宽度」hidden=units 做结构化剪枝。
#
# 设计要点
# ---------
# 1) RSSM 的 ``hidden``（与 ``model.units`` 一致，如 size100M 的 768）出现在：
#    - ``Deter``：``_dyn_in0/1/2`` 三个分支首层 Linear 的输出维必须同步缩维；随后
#      ``_dyn_hid`` 里第一个 ``BlockLinear`` 的输入维含 ``D/G + 3*H``（每块），
#      剪 H 时需删掉每块里三段 x0/x1/x2 的同一神经元下标。
#    - ``_obs_net`` / ``_img_net``：各隐藏层 Linear + RMSNorm 与 actor 剪枝同构。
# 2) ConvDecoder（仅 CNN 路径，Atari）：``sp1[0]`` 为 ``Linear(flat_stoch, 2*units)``，
#    与 RSSM 的 ``hidden`` 对齐的方式是按对 ``(2j, 2j+1)`` 剪掉通道，再同步 ``sp1``
#    的 RMSNorm 与 ``sp2`` 的输入维。
# 3) 不剪 ``deter`` / ``stoch`` 形状，``feat_size`` 不变，Actor/Value/Reward 等头无需改宽。
# 4) 同步 ``_frozen_rssm``（与 ``prune_actor_critic`` 里 frozen 镜像一致）。
#
# 依赖：Hydra / OmegaConf；**不**依赖 torch_pruning 建图（与 ``prune_actor_critic.py`` 相同哲学）。
#
# Colab：改 ``CHECKPOINT_PATH`` / ``SAVE_DIR`` / ``CONFIG_DIR``；本机若目录不存在会在 ``__main__`` 回落到仓库根目录。

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
from networks import BlockLinear, ConvDecoder

# ================================================================
# 路径与 Hydra
# ================================================================
CHECKPOINT_PATH = pathlib.Path(
    "/content/drive/MyDrive/r2dreamer_checkpoints/size100m_atari_breakout/latest.pt"
)
SAVE_DIR = pathlib.Path("/content/drive/MyDrive/r2dreamer_checkpoints/pruned_models_rssm_decoder")
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


def collect_data_and_compute_importance(agent: Dreamer, cfg, train_envs) -> torch.Tensor:
    """小 buffer + 多次 ``_cal_grad``，合并 RSSM 首层 hidden 与 Decoder ``sp1[0]`` 的 Taylor 重要性。"""
    if agent.rep_loss != "dreamer":
        raise RuntimeError("本脚本仅支持 rep_loss=dreamer（需 MultiDecoder._cnn）")
    if not hasattr(agent.decoder, "_cnn"):
        raise RuntimeError("当前 decoder 无 CNN 路径（检查 env.decoder.cnn_keys）")

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
        f"  累积 {CALIB_N_BATCHES} 个 batch 的梯度 (B={bs}, T={bl})..."
    )
    for i in range(CALIB_N_BATCHES):
        data, index, initial = replay_buffer.sample()
        data = data.to(cfg.device)
        initial = (initial[0].to(cfg.device), initial[1].to(cfg.device))
        data = agent.preprocess(data)
        agent._cal_grad(data, initial)
        if torch.cuda.is_available() and (i + 1) % 4 == 0:
            torch.cuda.empty_cache()

    dn = agent.rssm._deter_net
    H = dn._dyn_in0[0].out_features
    for name in ("_dyn_in0", "_dyn_in1", "_dyn_in2"):
        lin = getattr(dn, name)[0]
        if not isinstance(lin, nn.Linear) or lin.out_features != H:
            raise RuntimeError(f"Deter {name} 首层 Linear 异常: {type(lin)}, out={getattr(lin, 'out_features', None)}")

    def _taylor_out(lin: nn.Linear) -> torch.Tensor:
        g = lin.weight.grad
        if g is None:
            return torch.zeros(lin.out_features, device=lin.weight.device)
        return torch.abs(g * lin.weight).sum(dim=1).detach()

    acc = torch.zeros(H, device=agent.device)
    acc += _taylor_out(dn._dyn_in0[0])
    acc += _taylor_out(dn._dyn_in1[0])
    acc += _taylor_out(dn._dyn_in2[0])

    # _obs_net：第一个 Linear 输出维 = H
    obs_linears = [(n, m) for n, m in agent.rssm._obs_net.named_children() if isinstance(m, nn.Linear)]
    if not obs_linears:
        raise RuntimeError("_obs_net 中未找到 Linear")
    acc += _taylor_out(obs_linears[0][1])

    # _img_net：第一个 Linear
    img_linears = [(n, m) for n, m in agent.rssm._img_net.named_children() if isinstance(m, nn.Linear)]
    if not img_linears:
        raise RuntimeError("_img_net 中未找到 Linear")
    acc += _taylor_out(img_linears[0][1])

    # Decoder sp1[0]：输出 2*H，按对合并到 H 维再与 RSSM 同序打分
    sp1_lin0 = agent.decoder._cnn.sp1[0]
    if not isinstance(sp1_lin0, nn.Linear):
        raise RuntimeError("decoder._cnn.sp1[0] 应为 Linear")
    if sp1_lin0.out_features != 2 * H:
        raise RuntimeError(f"sp1[0] out_features={sp1_lin0.out_features} 与 2*H={2*H} 不一致")
    t2 = _taylor_out(sp1_lin0)
    pair = t2[0::2] + t2[1::2]
    acc += pair.to(acc.device)

    acc = acc.float().cpu()
    imp_min, imp_max = acc.min(), acc.max()
    acc = (acc - imp_min) / (imp_max - imp_min + 1e-8)
    print(f"  合并重要性完成，均值={acc.mean():.4f}")
    return acc


def _keep_indices(H: int, prune_idxs: list[int]) -> list[int]:
    rm = {int(i) for i in prune_idxs}
    keep = [i for i in range(H) if i not in rm]
    if not keep:
        raise RuntimeError("剪枝后无剩余 hidden 通道")
    return keep


def _prune_linear_out_in(
    lin: nn.Linear, keep: torch.Tensor, prune_out: bool, prune_in: bool
) -> None:
    W = lin.weight.data
    b = lin.bias.data if lin.bias is not None else None
    if prune_out and prune_in:
        Wn = W.index_select(1, keep).index_select(0, keep)
        bn = b.index_select(0, keep) if b is not None else None
    elif prune_out:
        Wn = W.index_select(0, keep)
        bn = b.index_select(0, keep) if b is not None else None
    elif prune_in:
        Wn = W.index_select(1, keep)
        bn = b
    else:
        return
    lin.weight = nn.Parameter(Wn.contiguous().clone())
    if bn is not None:
        lin.bias = nn.Parameter(bn.contiguous().clone())
    elif b is not None:
        lin.bias = None
    lin.in_features = Wn.shape[1]
    lin.out_features = Wn.shape[0]


def _prune_rmsnorm_channels(norm: nn.RMSNorm, keep: torch.Tensor) -> None:
    nw = norm.weight.data.index_select(0, keep)
    norm.weight = nn.Parameter(nw.contiguous().clone())
    norm.normalized_shape = (int(keep.numel()),)


def _prune_obs_net(seq: nn.Sequential, obs_layers: int, prune_idxs: list[int]) -> None:
    H = next(m.out_features for m in seq if isinstance(m, nn.Linear))
    keep_list = _keep_indices(H, prune_idxs)
    dev = next(seq.parameters()).device
    keep = torch.tensor(keep_list, dtype=torch.long, device=dev)

    for i in range(obs_layers):
        lin = getattr(seq, f"obs_net_{i}")
        norm = getattr(seq, f"obs_net_n_{i}")
        if not isinstance(lin, nn.Linear) or not isinstance(norm, nn.RMSNorm):
            raise RuntimeError(f"_obs_net 第{i}层结构异常")
        if i == 0:
            _prune_linear_out_in(lin, keep, prune_out=True, prune_in=False)
        else:
            _prune_linear_out_in(lin, keep, prune_out=True, prune_in=True)
        _prune_rmsnorm_channels(norm, keep)

    logit = getattr(seq, "obs_net_logit")
    if not isinstance(logit, nn.Linear):
        raise RuntimeError("缺少 obs_net_logit")
    _prune_linear_out_in(logit, keep, prune_out=False, prune_in=True)


def _prune_img_net(seq: nn.Sequential, img_layers: int, prune_idxs: list[int]) -> None:
    linears = [(n, m) for n, m in seq.named_children() if isinstance(m, nn.Linear)]
    if len(linears) != img_layers + 1:
        raise RuntimeError(f"_img_net Linear 数量 {len(linears)} 与 img_layers={img_layers} 不匹配")
    H = linears[0][1].out_features
    keep_list = _keep_indices(H, prune_idxs)
    dev = linears[0][1].weight.device
    keep = torch.tensor(keep_list, dtype=torch.long, device=dev)

    for i in range(img_layers):
        lin = getattr(seq, f"img_net_{i}")
        norm = getattr(seq, f"img_net_n_{i}")
        if not isinstance(lin, nn.Linear) or not isinstance(norm, nn.RMSNorm):
            raise RuntimeError(f"_img_net 第{i}层结构异常")
        if i == 0:
            _prune_linear_out_in(lin, keep, prune_out=True, prune_in=False)
        else:
            _prune_linear_out_in(lin, keep, prune_out=True, prune_in=True)
        _prune_rmsnorm_channels(norm, keep)

    logit = getattr(seq, "img_net_logit")
    if not isinstance(logit, nn.Linear):
        raise RuntimeError("缺少 img_net_logit")
    _prune_linear_out_in(logit, keep, prune_out=False, prune_in=True)


def _prune_dyn_in_branch(seq: nn.Sequential, keep: torch.Tensor) -> None:
    lin = seq[0]
    norm = seq[1]
    if not isinstance(lin, nn.Linear) or not isinstance(norm, nn.RMSNorm):
        raise RuntimeError("_dyn_in* 结构应为 Linear+RMSNorm+...")
    _prune_linear_out_in(lin, keep, prune_out=True, prune_in=False)
    _prune_rmsnorm_channels(norm, keep)


def _build_keep_in_per_block_deter(deter: int, blocks: int, H: int, prune_idxs: list[int]) -> list[int]:
    rm = {int(i) for i in prune_idxs}
    dg = deter // blocks
    keep: list[int] = list(range(dg))
    for k in range(3):
        base = dg + k * H
        for j in range(H):
            if j not in rm:
                keep.append(base + j)
    return keep


def _prune_deter_hidden(deter_mod, H: int, deter: int, blocks: int, prune_idxs: list[int]) -> int:
    """返回剪枝后的新 hidden 宽度 H'."""
    keep_list = _keep_indices(H, prune_idxs)
    dev = deter_mod._dyn_in0[0].weight.device
    keep = torch.tensor(keep_list, dtype=torch.long, device=dev)
    H_new = len(keep_list)

    for name in ("_dyn_in0", "_dyn_in1", "_dyn_in2"):
        _prune_dyn_in_branch(getattr(deter_mod, name), keep)

    # 第一个 dyn BlockLinear：缩输入维（每块 D/G + 3*H → D/G + 3*H'）
    bl0 = deter_mod._dyn_hid[0]

    if not isinstance(bl0, BlockLinear):
        raise RuntimeError("_dyn_hid[0] 应为 BlockLinear")
    keep_in_pb = _build_keep_in_per_block_deter(deter, blocks, H, prune_idxs)
    keep_in_t = torch.tensor(keep_in_pb, dtype=torch.long, device=dev)
    W = bl0.weight.data.index_select(1, keep_in_t)
    bl0.weight = nn.Parameter(W.contiguous().clone())
    bl0.in_ch = int(W.shape[1] * blocks)
    return H_new


def _prune_conv_decoder_sp_branch(dec: ConvDecoder, units: int, prune_idxs: list[int]) -> None:
    rm_flat: list[int] = []
    for j in sorted(set(int(x) for x in prune_idxs)):
        rm_flat.extend([2 * j, 2 * j + 1])
    rm_set = set(rm_flat)
    U2 = 2 * units
    keep_list = [i for i in range(U2) if i not in rm_set]
    if not keep_list:
        raise RuntimeError("Decoder sp1 剪枝后无剩余通道")
    dev = dec.sp1[0].weight.device
    keep = torch.tensor(keep_list, dtype=torch.long, device=dev)

    lin0 = dec.sp1[0]
    _prune_linear_out_in(lin0, keep, prune_out=True, prune_in=False)
    nrm = dec.sp1[1]
    if not isinstance(nrm, nn.RMSNorm):
        raise RuntimeError("sp1[1] 应为 RMSNorm")
    _prune_rmsnorm_channels(nrm, keep)

    _prune_linear_out_in(dec.sp2, keep, prune_out=False, prune_in=True)
    dec.units = dec.sp1[0].out_features // 2


def _apply_rssm_decoder_prune(agent: Dreamer, prune_idxs: list[int]) -> int:
    rssm = agent.rssm
    H = int(rssm._hidden)
    deter = int(rssm._deter)
    blocks = int(rssm._blocks)

    H_new = _prune_deter_hidden(rssm._deter_net, H, deter, blocks, prune_idxs)
    _prune_obs_net(rssm._obs_net, int(rssm._obs_layers), prune_idxs)
    _prune_img_net(rssm._img_net, int(rssm._img_layers), prune_idxs)
    rssm._hidden = H_new

    dec = agent.decoder._cnn
    if not isinstance(dec, ConvDecoder):
        raise RuntimeError("decoder._cnn 应为 ConvDecoder")
    _prune_conv_decoder_sp_branch(dec, int(dec.units), prune_idxs)

    # 冻结镜像
    fr = agent._frozen_rssm
    _prune_deter_hidden(fr._deter_net, H, deter, blocks, prune_idxs)
    _prune_obs_net(fr._obs_net, int(fr._obs_layers), prune_idxs)
    _prune_img_net(fr._img_net, int(fr._img_layers), prune_idxs)
    fr._hidden = H_new

    return H_new


def prune_rssm_decoder(agent: Dreamer, importance: torch.Tensor, prune_ratio: float) -> Dreamer:
    H = int(importance.numel())
    n_prune = int(H * prune_ratio)
    n_keep = H - n_prune
    _, sorted_indices = torch.sort(importance)
    pruning_idxs = sorted_indices[:n_prune].tolist()

    print(f"  总 hidden 通道: {H}, 剪掉: {n_prune}, 保留: {n_keep}")
    if n_prune > 0:
        print(f"  被删通道中最高重要性: {importance[sorted_indices[n_prune - 1]].item():.4f}")
        print(f"  被保留通道中最低重要性: {importance[sorted_indices[n_prune]].item():.4f}")

    H_new = _apply_rssm_decoder_prune(agent, pruning_idxs)
    print(f"  剪枝后 rssm._hidden = {H_new}；decoder._cnn.sp1[0] out = {agent.decoder._cnn.sp1[0].out_features}")
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
        initial = agent.rssm.initial(B)
        post_stoch, post_deter, _ = agent.rssm.observe(
            embed, data["action"], initial, data["is_first"]
        )
        _ = agent.decoder(post_stoch, post_deter)
        with torch.no_grad():
            _ = agent._frozen_rssm.observe(
                embed, data["action"], initial, data["is_first"]
            )
        print("  RSSM.observe + Decoder 前向成功（含 _frozen_rssm.observe）")
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

    prune_ratios = [0.2, 0.3, 0.4]

    for prune_ratio in prune_ratios:
        print(f"\n{'=' * 60}\n剪枝率: {prune_ratio * 100:.0f}%\n{'=' * 60}")

        cfg_model = copy.deepcopy(cfg_backup.model)
        print("加载原始模型...")
        agent = Dreamer(cfg_model, obs_space, act_space).to(cfg.device)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=cfg.device)
        agent.load_state_dict(ckpt["agent_state_dict"])

        print("计算重要性...")
        importance = collect_data_and_compute_importance(agent, cfg, train_envs)
        H0 = int(importance.numel())

        print("执行剪枝...")
        agent = prune_rssm_decoder(agent, importance, prune_ratio)

        print("验证...")
        if not verify_pruned_model(agent, obs_space, act_dim):
            print("  跳过保存")
            continue

        agent.float()
        H1 = int(agent.rssm._hidden)
        save_path = SAVE_DIR / f"rssm_decoder_pruned_{int(prune_ratio * 100)}pct.pt"
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "prune_ratio": prune_ratio,
                "pruned_modules": ["rssm", "decoder._cnn", "_frozen_rssm"],
                "rssm_hidden_before": H0,
                "rssm_hidden_after": H1,
            },
            save_path,
        )
        print(f"  已保存: {save_path}")

    print("\n全部完成。")


if __name__ == "__main__":
    if not pathlib.Path(CONFIG_DIR).exists():
        CONFIG_DIR = str(_REPO / "configs")
        CHECKPOINT_PATH = _REPO / "latest.pt"
        SAVE_DIR = _REPO / "pruned_models_rssm_decoder"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    main()
