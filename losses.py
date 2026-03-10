"""
复合损失函数模块

损失 = λ1 * L_intensity + λ2 * L_ssim + λ3 * L_gradient + λ4 * L_tv

各项含义:
  L_intensity : 强度保留（保证融合图亮度不低于 IR 和 VIS 的逐像素最大值）
  L_ssim      : 结构相似性（与可见光图像保持视觉结构一致）
  L_gradient  : 梯度保留（同时保留 IR 和 VIS 的边缘细节）
  L_tv        : 对齐平滑（对DCN偏移场施加总变分正则，防止跳跃伪影）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        # Load VGG16 features
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        blocks.append(vgg[:4].eval())
        blocks.append(vgg[4:9].eval())
        blocks.append(vgg[9:16].eval())
        blocks.append(vgg[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.mse_loss(x, y)
        return loss

# ──────────────────────────────────────────────────────────────
# 1. 强度损失 (Intensity Loss)
# ──────────────────────────────────────────────────────────────
def intensity_loss(fused: torch.Tensor,
                   ir: torch.Tensor,
                   vis: torch.Tensor) -> torch.Tensor:
    """
    强度保留损失。

    使用 max(IR, VIS) 作为亮度目标，同时对 target 做 3×3 均值平滑，
    强制网络输出空间连续的热量分布（消除孤立亮斑），同时保留 IR 发光效果。

    L_int = mean( |fused - smooth(max(ir, vis))| )
    """
    target = torch.max(ir, vis)
    # 9×9 均值平滑：更宽的光晕过渡区域，消除 IR 热区与 VIS 背景之间的"割裂感"
    target_smooth = F.avg_pool2d(target, kernel_size=9, stride=1, padding=4)
    return F.l1_loss(fused, target_smooth)


# ──────────────────────────────────────────────────────────────
# 2. 结构相似性损失（SSIM-based）
# ──────────────────────────────────────────────────────────────
def gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """生成 2D 高斯窗（用于 SSIM 计算）"""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel.unsqueeze(0).unsqueeze(0)


def ssim_loss(fused: torch.Tensor,
              target: torch.Tensor,
              window_size: int = 11) -> torch.Tensor:
    """
    单通道 SSIM 损失（1 - SSIM）。

    参数:
        fused  (Tensor): 融合图像 [B, 1, H, W]
        target (Tensor): 参考图像 [B, 1, H, W]

    返回:
        loss (Tensor): 标量，值域 [0, 2]（完全不相似=2）
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    kernel = gaussian_kernel(window_size).to(fused.device)
    pad = window_size // 2

    mu1 = F.conv2d(fused, kernel, padding=pad)
    mu2 = F.conv2d(target, kernel, padding=pad)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(fused ** 2, kernel, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, kernel, padding=pad) - mu2_sq
    sigma12 = F.conv2d(fused * target, kernel, padding=pad) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


# ──────────────────────────────────────────────────────────────
# 3. 梯度保留损失
# ──────────────────────────────────────────────────────────────
def gradient_loss(fused: torch.Tensor,
                  ir: torch.Tensor,
                  vis: torch.Tensor) -> torch.Tensor:
    """
    Sobel 梯度保留损失。

    要求融合图像的梯度幅值不低于 IR 和 VIS 各自梯度的最大值，
    从而同时保留两路图像的边缘信息。

    L_grad = mean( |grad(fused) - max(grad(ir), grad(vis))| )
    """
    def sobel_gradient(x: torch.Tensor) -> torch.Tensor:
        """计算 Sobel 梯度幅值 [B, 1, H, W]"""
        # Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    grad_fused = sobel_gradient(fused)
    grad_ir = sobel_gradient(ir)
    grad_vis = sobel_gradient(vis)
    grad_target = torch.max(grad_ir, grad_vis)

    return F.l1_loss(grad_fused, grad_target)


# ──────────────────────────────────────────────────────────────
# 4. 归一化互相关（NCC）对齐损失
# ──────────────────────────────────────────────────────────────
def ncc_loss(aligned_ir: torch.Tensor,
             vis: torch.Tensor,
             window_size: int = 9) -> torch.Tensor:
    """
    局部归一化互相关（NCC）损失，用于自监督图像配准。

    NCC = cov(ir, vis) / (std(ir) * std(vis))
    完全对齐时 NCC=1，损失=0。

    优点：对光度差异（红外热图 vs 可见光亮度）鲁棒，
    只要结构/边缘对应即可，不要求绝对强度一致。
    能真正驱动 DCN 向正确位置移动，而不只是令梯度幅值接近。
    """
    B, C, H, W = aligned_ir.shape
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size,
                        device=aligned_ir.device) / (window_size ** 2)

    def local_stats(x):
        mu = F.conv2d(x, kernel, padding=pad)
        var = F.conv2d(x ** 2, kernel, padding=pad) - mu ** 2
        return mu, var.clamp(min=1e-6)

    mu1, var1 = local_stats(aligned_ir)
    mu2, var2 = local_stats(vis)
    cov = F.conv2d(aligned_ir * vis, kernel, padding=pad) - mu1 * mu2
    ncc = cov / (torch.sqrt(var1) * torch.sqrt(var2) + 1e-8)
    # 限制在[-1,1]内再取均值，避免数值不稳定
    ncc = ncc.clamp(-1.0, 1.0)
    return 1.0 - ncc.mean()


# ──────────────────────────────────────────────────────────────
# 5. 总变分平滑损失（用于对齐偏移场正则化）
# ──────────────────────────────────────────────────────────────
def tv_loss(tensor: torch.Tensor) -> torch.Tensor:
    """
    总变分（Total Variation）损失。
    对偏移场或图像施加平滑约束，防止产生跳跃不连续的形变。

    参数:
        tensor (Tensor): [B, C, H, W]（通常是 DCN 的 offset map）

    返回:
        loss (Tensor): 标量
    """
    diff_h = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).mean()
    diff_w = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).mean()
    return diff_h + diff_w


# ──────────────────────────────────────────────────────────────
# 6. 复合总损失
# ──────────────────────────────────────────────────────────────
def fusion_loss(
    fused: torch.Tensor,
    aligned_ir: torch.Tensor,
    vis: torch.Tensor,
    lambda_int: float = 1.0,
    lambda_ssim: float = 1.0,
    lambda_grad: float = 0.5,
    lambda_tv: float = 0.0,    
    offset: torch.Tensor = None,
    lambda_align: float = 5.0,
    vgg_loss: torch.nn.Module = None,
    lambda_perceptual: float = 1.0,
) -> tuple:
    """
    复合融合损失函数。
    """
    def sobel_gradient(x: torch.Tensor) -> torch.Tensor:
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    # 1. 常规融合损失
    L_int = intensity_loss(fused, aligned_ir, vis)
    L_ssim_ir = ssim_loss(fused, aligned_ir)
    L_ssim_vis = ssim_loss(fused, vis)
    L_ssim = (L_ssim_ir + L_ssim_vis) / 2.0
    
    # 梯度保留损失
    grad_fused = sobel_gradient(fused)
    grad_ir = sobel_gradient(aligned_ir)
    grad_vis = sobel_gradient(vis)
    grad_target = torch.max(grad_ir, grad_vis)
    L_grad = F.l1_loss(grad_fused, grad_target)

    # 2. 跨模态自监督对齐损失（NCC）
    L_align = ncc_loss(aligned_ir, vis)

    total = lambda_int * L_int + lambda_ssim * L_ssim + lambda_grad * L_grad + lambda_align * L_align

    # 3. 感知损失 (Perceptual Loss)
    L_perceptual = torch.tensor(0.0, device=fused.device)
    if vgg_loss is not None:
        L_perceptual = vgg_loss(fused, vis)
        total = total + lambda_perceptual * L_perceptual

    # 4. TV 平滑约束
    L_tv = torch.tensor(0.0, device=fused.device)
    if lambda_tv > 0 and offset is not None:
        L_tv = tv_loss(offset)
        total = total + lambda_tv * L_tv

    detail = {
        'total': total.item(),
        'intensity': L_int.item(),
        'ssim': L_ssim.item(),
        'gradient': L_grad.item(),
        'align': L_align.item(),
        'perceptual': L_perceptual.item(),
        'tv': L_tv.item(),
    }

    return total, detail
