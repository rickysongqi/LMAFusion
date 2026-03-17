"""
复合损失函数模块

损失 = λ1 * L_intensity + λ2 * L_ssim + λ3 * L_gradient + λ4 * L_tv

各项含义:
  L_intensity : 强度保留（保证融合图亮度不低于 IR 和 VIS 的逐像素最大值）+ 红外显著区域增强
  L_ssim      : 结构相似性（与红外和可见光图像保持视觉结构一致，红外权重更高）
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

    def _preprocess(self, img):
        if img.shape[1] != 3:
            img = img.repeat(1, 3, 1, 1)
        img = (img - self.mean) / self.std
        if self.resize:
            img = self.transform(img, mode='bilinear', size=(224, 224), align_corners=False)
        return img

    def _extract_features(self, img):
        feats = []
        x = img
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats

    def forward(self, input, target):
        x = self._preprocess(input)
        y = self._preprocess(target)
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.mse_loss(x, y)
        return loss

    def forward_symmetric(self, fused, ir, vis):
        """对称感知损失：fused 特征只提取一次，分别与 IR 和 VIS 对比"""
        f = self._extract_features(self._preprocess(fused))
        f_ir = self._extract_features(self._preprocess(ir))
        f_vis = self._extract_features(self._preprocess(vis))
        loss_ir = sum(F.mse_loss(a, b) for a, b in zip(f, f_ir))
        loss_vis = sum(F.mse_loss(a, b) for a, b in zip(f, f_vis))
        return 0.5 * loss_ir + 0.5 * loss_vis

# ──────────────────────────────────────────────────────────────
# 1. 强度损失 (Intensity Loss) — 含红外显著区域增强
# ──────────────────────────────────────────────────────────────
def intensity_loss(fused: torch.Tensor,
                   ir: torch.Tensor,
                   vis: torch.Tensor) -> torch.Tensor:
    """
    强度保留损失：逐像素取 max(IR, VIS) 作为目标。

    max 策略天然适配白天/夜间场景：
      - 白天（CSMR）：VIS 整体较亮时自动跟随 VIS，IR 局部热辐射更亮处保留 IR
      - 夜间（MSRS）：IR 热目标高亮处保留 IR，VIS 背景亮处保留 VIS
    """
    target = torch.max(ir, vis)
    return F.l1_loss(fused, target)


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
# 3. Spatial Frequency (SF) 纹理保留损失
# ──────────────────────────────────────────────────────────────
def spatial_frequency(x: torch.Tensor, kernel_radius: int = 5) -> torch.Tensor:
    """
    计算图像的局部空间频率 (Spatial Frequency)。

    SF 衡量局部区域的纹理/细节丰富程度，比简单 Sobel 梯度更能
    捕捉红外热图中细腻分散的热对比信息。

    参数:
        x (Tensor): [B, 1, H, W]
        kernel_radius (int): 局部统计窗口半径

    返回:
        sf_map (Tensor): [B, 1, H, W] 局部空间频率图
    """
    B, C, H, W = x.shape
    r_shift_kernel = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                                  dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
    b_shift_kernel = torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                                  dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)

    x_r_shift = F.conv2d(x, r_shift_kernel, padding=1)
    x_b_shift = F.conv2d(x, b_shift_kernel, padding=1)

    x_grad = (x_r_shift - x) ** 2 + (x_b_shift - x) ** 2

    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones(1, 1, kernel_size, kernel_size,
                            dtype=x.dtype, device=x.device)
    kernel_padding = kernel_size // 2
    x_sf = F.conv2d(x_grad, add_kernel, padding=kernel_padding)
    return x_sf


def sf_loss(fused: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    空间频率损失：要求融合图的局部纹理丰富度与参考图一致。

    L_sf = ||SF(fused) - SF(target)||_2
    """
    return torch.norm(spatial_frequency(fused) - spatial_frequency(target))


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
# 4b. 边缘域 NCC 对齐损失（跨模态更鲁棒）
# ──────────────────────────────────────────────────────────────
def edge_ncc_loss(aligned_ir: torch.Tensor,
                  vis: torch.Tensor,
                  window_size: int = 9) -> torch.Tensor:
    """
    在 Sobel 边缘域上计算 NCC，比原始像素域更适合跨模态对齐。

    IR 和 VIS 的原始像素分布差异巨大（热图 vs 亮度图），
    但两者的边缘/轮廓在空间上是同构的。先提取边缘再算 NCC，
    可以获得更稳定的对齐梯度信号。
    """
    def sobel_edges(x):
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=x.dtype, device=x.device).reshape(1, 1, 3, 3)
        return torch.sqrt(F.conv2d(x, sx, padding=1)**2 +
                          F.conv2d(x, sy, padding=1)**2 + 1e-8)

    edge_ir = sobel_edges(aligned_ir)
    edge_vis = sobel_edges(vis)
    return ncc_loss(edge_ir, edge_vis, window_size)


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
def guidance_loss(fused: torch.Tensor,
                  guidance: torch.Tensor) -> torch.Tensor:
    """
    自适应引导损失：SSIM + SF 约束 fused 向引导图像靠拢。
    引导图像是模型自身历史最佳输出 (I2)。
    """
    L_ssim = ssim_loss(fused, guidance)
    L_sf = 1e-5 * sf_loss(fused, guidance)
    return L_ssim + L_sf


def fusion_loss(
    fused: torch.Tensor,
    aligned_ir: torch.Tensor,
    vis: torch.Tensor,
    lambda_int: float = 1.0,
    lambda_ssim: float = 1.0,
    lambda_sf: float = 1.0,
    lambda_tv: float = 0.0,
    offset: torch.Tensor = None,
    lambda_align: float = 5.0,
    vgg_loss: torch.nn.Module = None,
    lambda_perceptual: float = 1.0,
    use_align: bool = False,
    guidance_img: torch.Tensor = None,
    guidance_weight: float = 0.0,
) -> tuple:
    """
    复合融合损失函数。

    内容损失 = 强度 + SSIM + SF（替代 Sobel 梯度） + 感知 + 对齐 + TV
    引导损失 = guidance_weight * (SSIM + SF)(fused, guidance_img)
    """
    # 1. 强度损失
    L_int = intensity_loss(fused, aligned_ir, vis)

    # 2. SSIM 损失：均衡保留双模态结构
    L_ssim_ir = ssim_loss(fused, aligned_ir)
    L_ssim_vis = ssim_loss(fused, vis)
    L_ssim = 0.5 * L_ssim_ir + 0.5 * L_ssim_vis

    # 3. SF 纹理保留损失（替代 Sobel 梯度）
    L_sf_ir = 1e-5 * sf_loss(fused, aligned_ir)
    L_sf_vis = 1e-5 * sf_loss(fused, vis)
    L_sf = L_sf_ir + L_sf_vis

    total = lambda_int * L_int + lambda_ssim * L_ssim + lambda_sf * L_sf

    # 4. 对齐损失
    L_align = torch.tensor(0.0, device=fused.device)
    if use_align:
        L_align = edge_ncc_loss(aligned_ir, vis)
        total = total + lambda_align * L_align

    # 5. 感知损失
    L_perceptual = torch.tensor(0.0, device=fused.device)
    if vgg_loss is not None:
        L_perceptual = vgg_loss.forward_symmetric(fused, aligned_ir, vis)
        total = total + lambda_perceptual * L_perceptual

    # 6. TV 平滑约束
    L_tv = torch.tensor(0.0, device=fused.device)
    if lambda_tv > 0 and offset is not None:
        L_tv = tv_loss(offset)
        total = total + lambda_tv * L_tv

    # 7. 自适应引导损失（仅在 guidance_weight > 0 且 guidance_img 有效时启用）
    L_guidance = torch.tensor(0.0, device=fused.device)
    if guidance_weight > 0 and guidance_img is not None:
        L_guidance = guidance_loss(fused, guidance_img)
        total = total + guidance_weight * L_guidance

    detail = {
        'total': total.item(),
        'intensity': L_int.item(),
        'ssim': L_ssim.item(),
        'sf': L_sf.item(),
        'align': L_align.item(),
        'perceptual': L_perceptual.item(),
        'tv': L_tv.item(),
        'guidance': L_guidance.item(),
    }

    return total, detail
