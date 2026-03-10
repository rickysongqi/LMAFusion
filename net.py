"""
LMAFusion 主网络：Lightweight Mamba-guided Adaptive Fusion

网络架构：
  输入(IR, VIS)
    → [DCN对齐] → 对齐后IR + VIS
    → [双路浅层CNN编码]
    → [通道交换] (零FLOPs跨模态渗透)
    → [双路Mamba深层编码]
    → [双向门控融合]
    → [解码器] → 融合图像 [B, 1, H, W]

参数量目标: < 200K
"""

import torch
import torch.nn as nn
from modules import (
    DeformableAlignment,
    ChannelExchange,
    SimpleMambaBlock,
    BidirectionalGating,
)
from modules.bidirectional_gating import FusionDecoder


class ShallowEncoder(nn.Module):
    """
    双路浅层 CNN 编码器（IR 或 VIS 各一个实例）。

    3层结构：
      Conv(1→base_channels, 3×3) → BN → ReLU
      Conv(base→base, 3×3, groups=base) → BN → ReLU  [深度可分离]
      Conv(base→base*2, 1×1) → BN → ReLU              [通道扩展]

    内置三个 BN 层，供 ChannelExchange 读取 γ 值。

    参数:
        in_ch   : 输入通道（灰度图像 = 1）
        base_ch : 基础通道数（默认 16）
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 16):
        super().__init__()
        # 第1层：标准卷积
        self.conv1 = nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_ch)
        self.relu1 = nn.ReLU(inplace=True)

        # 第2层：深度可分离卷积（参数高效）
        self.conv2_dw = nn.Conv2d(base_ch, base_ch, 3, padding=1, groups=base_ch, bias=False)
        self.conv2_pw = nn.Conv2d(base_ch, base_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_ch)
        self.relu2 = nn.ReLU(inplace=True)

        # 第3层：1×1 通道扩展（用于通道交换交互）
        self.conv3 = nn.Conv2d(base_ch, base_ch * 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(base_ch * 2)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        返回:
            feat (Tensor): 浅层特征 [B, base_ch*2, H, W]
            bn_for_exchange (nn.BatchNorm2d): 用于通道交换的 BN 层（第2层）
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2_pw(self.conv2_dw(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x, self.bn3   # 返回最后一层 BN 供交换参考


class LMAFusion(nn.Module):
    """
    LMAFusion 主网络。

    参数:
        base_ch   (int): 基础通道数（默认 16，总参数量约 150K）
        d_state   (int): Mamba SSM 隐状态维度（默认 16）
        exchange_p(float): 通道交换比例（默认 0.5）
        align_mid_ch(int): 对齐模块中间通道（默认 16）
    """

    def __init__(
        self,
        base_ch: int = 16,
        d_state: int = 16,
        exchange_p: float = 0.5,
        align_mid_ch: int = 16,
    ):
        super().__init__()
        self.base_ch = base_ch
        shallow_out_ch = base_ch * 2  # 浅层编码后通道数

        # ── 模块1：DCN 形变对齐 ────────────────────────────────────
        self.align = DeformableAlignment(in_channels=1, mid_channels=align_mid_ch)

        # ── 模块2：浅层编码器（各一个，不共享权重） ──────────────────
        self.ir_encoder = ShallowEncoder(in_ch=1, base_ch=base_ch)
        self.vis_encoder = ShallowEncoder(in_ch=1, base_ch=base_ch)

        # ── 模块3：通道交换层（零参数） ───────────────────────────────
        self.channel_exchange = ChannelExchange(p=exchange_p)

        # ── 模块4：Mamba 深层编码器（各一个，不共享权重） ─────────────
        self.ir_mamba = SimpleMambaBlock(dim=shallow_out_ch, d_state=d_state, expand=2)
        self.vis_mamba = SimpleMambaBlock(dim=shallow_out_ch, d_state=d_state, expand=2)

        # ── 模块5：双向门控融合 ────────────────────────────────────
        self.bidirectional_gate = BidirectionalGating(channels=shallow_out_ch)

        # ── 模块6：融合解码器 ──────────────────────────────────────
        self.decoder = FusionDecoder(in_channels=shallow_out_ch, mid_channels=base_ch)

    def forward(self, ir: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            ir  (Tensor): 红外图像 [B, 1, H, W]，值域 [0, 1]
            vis (Tensor): 可见光图像 [B, 1, H, W]，值域 [0, 1]

        返回:
            fused (Tensor): 融合图像 [B, 1, H, W]，值域 [0, 1]
        """
        # ── Step1: 形变对齐（IR 向 VIS 对齐） ──────────────────────
        ir_aligned, vis_ref, offset = self.align(ir, vis)

        # 用于融合的对齐图
        ir_aligned_for_fusion = ir_aligned

        # ── Step2: 双路浅层编码 ─────────────────────────────────────
        feat_ir, bn_ir = self.ir_encoder(ir_aligned_for_fusion)    # [B, 2*base_ch, H, W]
        feat_vis, bn_vis = self.vis_encoder(vis_ref)    # [B, 2*base_ch, H, W]

        # ── Step3: 通道交换（零 FLOPs 跨模态浅层渗透） ────────────────
        feat_ir, feat_vis = self.channel_exchange(feat_ir, feat_vis, bn_ir, bn_vis)

        # ── Step4: Mamba 深层全局提取 ──────────────────────────────
        feat_ir_deep = self.ir_mamba(feat_ir)           # [B, 2*base_ch, H, W]
        feat_vis_deep = self.vis_mamba(feat_vis)        # [B, 2*base_ch, H, W]

        # ── Step5: 双向门控融合 ────────────────────────────────────
        fused_feat = self.bidirectional_gate(feat_ir_deep, feat_vis_deep)

        # ── Step6: 解码重建 → 输出融合图像 ────────────────────────
        fused = self.decoder(fused_feat)

        if self.training:
            return fused, ir_aligned, offset
        return fused


def count_parameters(model: nn.Module) -> dict:
    """统计模型各模块参数量"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_params[name] = params
    module_params['total'] = total
    return module_params


if __name__ == '__main__':
    import torch

    print("=" * 60)
    print("LMAFusion 网络前向传播测试")
    print("=" * 60)

    model = LMAFusion(base_ch=16, d_state=16)
    model.eval()

    # 模拟 640×512 无人机前端分辨率
    B, H, W = 1, 256, 256
    ir_dummy = torch.randn(B, 1, H, W)
    vis_dummy = torch.randn(B, 1, H, W)

    with torch.no_grad():
        output = model(ir_dummy, vis_dummy)

    print(f"输入 IR  : {ir_dummy.shape}")
    print(f"输入 VIS : {vis_dummy.shape}")
    print(f"输出融合 : {output.shape}")
    print(f"输出值域 : [{output.min():.4f}, {output.max():.4f}]")

    # 参数量统计
    params = count_parameters(model)
    print("\n各模块参数量：")
    for name, cnt in params.items():
        if name != 'total':
            print(f"  {name:25s}: {cnt:,} ({cnt/1e3:.1f}K)")
    print(f"  {'总参数量':25s}: {params['total']:,} ({params['total']/1e3:.1f}K)")

    # 尝试 thop 测量 FLOPs
    try:
        from thop import profile, clever_format
        flops, params_cnt = profile(model, inputs=(ir_dummy, vis_dummy), verbose=False)
        flops, params_str = clever_format([flops, params_cnt], "%.3f")
        print(f"\nGFLOPs : {flops}")
        print(f"参数量 : {params_str}")
    except ImportError:
        print("\n(安装 thop 可自动统计 GFLOPs: pip install thop)")
