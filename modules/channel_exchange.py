"""
模块2：跨模态通道交换机制（Cross-modal Channel Exchange）

利用 BatchNorm 的缩放因子 γ 作为通道活跃度度量，
对低活跃度（信息冗余）通道进行跨模态对调，
实现近零 FLOPs 的模态间底层信息渗透。
"""

import torch
import torch.nn as nn


class ChannelExchange(nn.Module):
    """
    跨模态通道交换模块。

    通过比较两个 BatchNorm 层的 γ（缩放因子）绝对值，
    找到信息冗余的通道并在 IR 和 VIS 特征之间进行对调，
    使两路特征在浅层就产生跨模态"渗透"效应。

    参数:
        p (float): 参与交换的通道比例（默认 1/2，即交换活跃度最低的 50% 通道）
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self,
                feat_ir: torch.Tensor,
                feat_vis: torch.Tensor,
                bn_ir: nn.BatchNorm2d,
                bn_vis: nn.BatchNorm2d) -> tuple:
        """
        前向传播。

        参数:
            feat_ir  (Tensor): 红外浅层特征 [B, C, H, W]
            feat_vis (Tensor): 可见光浅层特征 [B, C, H, W]
            bn_ir    (BatchNorm2d): 红外编码器中对应的 BN 层
            bn_vis   (BatchNorm2d): 可见光编码器中对应的 BN 层

        返回:
            feat_ir_out  (Tensor): 注入了可见光信息的红外特征
            feat_vis_out (Tensor): 注入了红外信息的可见光特征
        """
        # 获取 BatchNorm 的缩放因子 γ：形状 [C]
        gamma_ir = bn_ir.weight.detach().abs()    # 红外各通道活跃度
        gamma_vis = bn_vis.weight.detach().abs()  # 可见光各通道活跃度

        # 计算每个通道两侧综合活跃度（取几何平均，更稳定）
        combined_activity = (gamma_ir * gamma_vis).sqrt()

        # 找出活跃度最低的 p 比例通道索引
        num_channels = feat_ir.shape[1]
        num_exchange = max(1, int(num_channels * self.p))
        _, inactive_idx = torch.topk(combined_activity, k=num_exchange, largest=False)

        # 直接对调对应通道：无需任何 matmul 或卷积
        feat_ir_out = feat_ir.clone()
        feat_vis_out = feat_vis.clone()
        feat_ir_out[:, inactive_idx] = feat_vis[:, inactive_idx]
        feat_vis_out[:, inactive_idx] = feat_ir[:, inactive_idx]

        return feat_ir_out, feat_vis_out


class ChannelExchangeSimple(nn.Module):
    """
    简化版通道交换：固定交换前 p 比例通道（不依赖 BN 层的 gamma）。
    用于调试或 BN gamma 访问困难时的替代方案。

    参数:
        p (float): 固定交换的通道比例（默认 1/2）
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, feat_ir: torch.Tensor, feat_vis: torch.Tensor) -> tuple:
        """
        前向传播。

        参数:
            feat_ir  (Tensor): 红外浅层特征 [B, C, H, W]
            feat_vis (Tensor): 可见光浅层特征 [B, C, H, W]

        返回:
            (Tensor, Tensor): 交换后的 (IR特征, VIS特征)
        """
        B, C, H, W = feat_ir.shape
        num_exchange = max(1, int(C * self.p))

        feat_ir_out = feat_ir.clone()
        feat_vis_out = feat_vis.clone()

        # 固定交换前 num_exchange 个通道
        feat_ir_out[:, :num_exchange] = feat_vis[:, :num_exchange]
        feat_vis_out[:, :num_exchange] = feat_ir[:, :num_exchange]

        return feat_ir_out, feat_vis_out
