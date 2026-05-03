# pruning_utils.py
import torch.nn as nn
import torch_pruning as tp
from networks import BlockLinear


class RMSNormPruningHandler(tp.pruner.BasePruningFunc):
    TARGET_MODULES = nn.RMSNorm

    def prune_out_channels(self, layer: nn.RMSNorm, idxs: list):
        keep_idxs = [i for i in range(layer.normalized_shape[0]) if i not in idxs]
        layer.weight = nn.Parameter(layer.weight.data[keep_idxs])
        layer.normalized_shape = (len(keep_idxs),)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer: nn.RMSNorm):
        return layer.normalized_shape[0]

    def get_in_channels(self, layer: nn.RMSNorm):
        return layer.normalized_shape[0]

class BlockLinearPruningHandler(tp.pruner.BasePruningFunc):
    TARGET_MODULES = BlockLinear

    def prune_out_channels(self, layer: BlockLinear, idxs: list):
        """
        idxs是要删除的输出通道索引（相对于flat的out_ch维度）
        需要把flat索引转换成block内的位置索引
        """
        blocks = layer.blocks
        out_per_block = layer.out_ch // blocks

        # 把flat索引转换成block内位置索引
        # 例如 out_ch=8, blocks=2, out_per_block=4
        # flat索引0→block内位置0, flat索引1→block内位置1
        # flat索引4→block内位置0（第二个block）
        # 我们只关心block内的位置，不关心是哪个block
        inner_idxs_to_remove = set(i % out_per_block for i in idxs)

        # 验证：必须是所有block都删相同位置（合法剪枝的必要条件）
        # 即idxs里每个block内被删的位置集合必须完全相同
        for b in range(blocks):
            block_idxs = set(
                i % out_per_block
                for i in idxs
                if i // out_per_block == b
            )
            if block_idxs and block_idxs != inner_idxs_to_remove:
                raise ValueError(
                    f"BlockLinear剪枝必须在每个block内删相同位置，"
                    f"但block{b}要删{block_idxs}，"
                    f"与其他block的{inner_idxs_to_remove}不一致"
                )

        # 计算保留的block内位置
        keep_inner = [
            i for i in range(out_per_block)
            if i not in inner_idxs_to_remove
        ]
        new_out_per_block = len(keep_inner)

        # 修改weight: (out//G, in//G, G) → (new_out//G, in//G, G)
        layer.weight = nn.Parameter(layer.weight.data[keep_inner, :, :])

        # 修改bias: (out_ch,) → (new_out_ch,)
        # bias是flat的，需要保留每个block里keep_inner对应的位置
        keep_bias_idxs = [
            b * out_per_block + i
            for b in range(blocks)
            for i in keep_inner
        ]
        layer.bias = nn.Parameter(layer.bias.data[keep_bias_idxs])

        # 更新维度属性
        layer.out_ch = new_out_per_block * blocks
        return layer

    def prune_in_channels(self, layer: BlockLinear, idxs: list):
        """
        剪输入通道，修改weight的第1维
        """
        blocks = layer.blocks
        in_per_block = layer.in_ch // blocks

        inner_idxs_to_remove = set(i % in_per_block for i in idxs)
        keep_inner = [
            i for i in range(in_per_block)
            if i not in inner_idxs_to_remove
        ]

        # 修改weight: (out//G, in//G, G) → (out//G, new_in//G, G)
        layer.weight = nn.Parameter(layer.weight.data[:, keep_inner, :])

        # 更新维度属性
        layer.in_ch = len(keep_inner) * blocks
        return layer

    def get_out_channels(self, layer: BlockLinear):
        return layer.out_ch

    def get_in_channels(self, layer: BlockLinear):
        return layer.in_ch