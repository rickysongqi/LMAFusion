"""
C2F 跨模态图像配准工具（Coarse-to-Fine）

对未对齐的红外/可见光图像对进行粗到精配准：
  - Coarse: 基于边缘特征的全局单应性（Homography）配准
  - Fine:   基于 ECC（Enhanced Correlation Coefficient）的亚像素精配准

用法:
  # 命令行模式
  python align_c2f.py --ir_dir ./unaligned/ir --vis_dir ./unaligned/vis --output_dir ./aligned

  # GUI 模式
  python align_c2f.py --gui
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ── 核心配准函数 ──────────────────────────────────────────────

def is_night_image(gray, osd_top_ratio=0.12, osd_bot_ratio=0.05, thresh=60):
    """
    自动判断图像是否为夜间场景。
    取图像中心有效区域（排除 OSD 文字）的均值亮度进行判断。

    参数:
        gray       : 灰度图 numpy 数组
        thresh     : 均值亮度阈值，低于此值视为夜间（默认 60）
    返回:
        bool — True 表示夜间，False 表示白天
    """
    h, w = gray.shape
    top_h = int(h * osd_top_ratio)
    bot_h = int(h * osd_bot_ratio)
    roi = gray[top_h: h - bot_h if bot_h > 0 else h, :]
    return float(roi.mean()) < thresh


def create_osd_mask(h, w, top_ratio=0.12, bottom_ratio=0.05):
    """
    生成 OSD 遮挡掩膜：屏蔽顶部文字和底部信息条。
    mask=255 表示有效区域，mask=0 表示屏蔽区域。
    """
    mask = np.ones((h, w), dtype=np.uint8) * 255
    top_h = int(h * top_ratio)
    bot_h = int(h * bottom_ratio)
    mask[:top_h, :] = 0
    if bot_h > 0:
        mask[-bot_h:, :] = 0
    return mask


def extract_edge_features(img_gray, max_keypoints=2000, mask=None):
    """
    在 CLAHE 增强 + 边缘图上提取 ORB 特征点。
    CLAHE 增强微弱纹理，边缘域对 IR/VIS 模态差异鲁棒。
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    edges = cv2.Canny(enhanced, 30, 120)
    edges = cv2.dilate(edges, None, iterations=1)

    if mask is not None:
        edges = cv2.bitwise_and(edges, mask)

    orb = cv2.ORB_create(nfeatures=max_keypoints, scaleFactor=1.2, nlevels=12)
    kp, des = orb.detectAndCompute(enhanced, mask)
    return kp, des, edges


def match_features(des_ir, des_vis, ratio_thresh=0.75):
    """
    使用 BFMatcher + Lowe's ratio test 进行特征匹配。
    """
    if des_ir is None or des_vis is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(des_ir, des_vis, k=2)
    except cv2.error:
        return []

    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
    return good


def estimate_similarity(kp_ir, kp_vis, good_matches, reproj_thresh=5.0):
    """
    RANSAC 估计 4 参数相似变换（平移 + 均匀缩放 + 旋转）。
    比 8 参数单应性更稳定，适合双相机（不同焦距）对同一目标成像的场景。
    返回 2x3 仿射矩阵。
    """
    if len(good_matches) < 6:
        return None, None

    pts_ir = np.float32([kp_ir[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_vis = np.float32([kp_vis[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffinePartial2D(
        pts_ir, pts_vis, method=cv2.RANSAC, ransacReprojThreshold=reproj_thresh
    )
    if M is None:
        return None, None

    a, b = M[0, 0], M[0, 1]
    scale = np.sqrt(a * a + b * b)
    angle_deg = np.abs(np.degrees(np.arctan2(b, a)))

    if scale < 0.7 or scale > 1.4:
        return None, mask
    if angle_deg > 15:
        return None, mask

    if mask is not None:
        inlier_ratio = mask.ravel().sum() / len(mask)
        if inlier_ratio < 0.2:
            return None, mask

    return M, mask


def refine_ecc(ir_gray, vis_gray, H_init, max_iter=200, eps=1e-5):
    """
    ECC 亚像素精配准：在粗配准基础上优化仿射/单应性参数。
    使用仿射模型（6 参数）比完整单应性（8 参数）更稳定。
    """
    h, w = vis_gray.shape
    if H_init is not None:
        warp_init = H_init[:2, :].astype(np.float32)
    else:
        warp_init = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    try:
        _, warp_matrix = cv2.findTransformECC(
            vis_gray, ir_gray, warp_init, cv2.MOTION_AFFINE, criteria,
            inputMask=None, gaussFiltSize=5
        )
        return warp_matrix
    except cv2.error:
        return warp_init


def _fill_border(aligned_ir, ir_img_resized, vis_img, warp, M_coarse, h, w):
    """用 VIS 像素填充 warp 后的无效区域（黑边）。"""
    white = np.ones_like(ir_img_resized, dtype=np.uint8) * 255
    if M_coarse is not None and warp is not None:
        white = cv2.warpAffine(white, M_coarse, (w, h))
        white = cv2.warpAffine(white, warp, (w, h))
    elif M_coarse is not None:
        white = cv2.warpAffine(white, M_coarse, (w, h))
    elif warp is not None:
        white = cv2.warpAffine(white, warp, (w, h))

    valid_mask = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY) > 128
    vis_fill = cv2.resize(vis_img, (w, h)) if vis_img.shape[:2] != (h, w) else vis_img
    aligned_ir[~valid_mask] = vis_fill[~valid_mask]


def align_single_pair(ir_path: str, vis_path: str, use_fine: bool = True,
                      night_thresh: int = 60):
    """
    对单对 IR/VIS 图像执行 C2F 配准。

    返回:
        aligned_ir: 配准后的 IR 图像（与 VIS 尺寸一致）
        H: 粗配准单应性矩阵
        warp: 精配准仿射矩阵
        n_matches: 匹配的特征点数
    """
    ir_buf = np.fromfile(ir_path, dtype=np.uint8)
    vis_buf = np.fromfile(vis_path, dtype=np.uint8)
    ir_img = cv2.imdecode(ir_buf, cv2.IMREAD_COLOR)
    vis_img = cv2.imdecode(vis_buf, cv2.IMREAD_COLOR)

    if ir_img is None:
        raise IOError(f"无法读取 IR 图像: {ir_path}")
    if vis_img is None:
        raise IOError(f"无法读取 VIS 图像: {vis_path}")

    ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
    vis_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)

    h, w = vis_gray.shape
    ir_h, ir_w = ir_gray.shape

    # ── 自动昼/夜判断（取 VIS 中心区域均值亮度）────────────────────
    night_mode = is_night_image(vis_gray, thresh=night_thresh)

    # 先将 IR 缩放到 VIS 尺寸
    if (ir_h, ir_w) != (h, w):
        ir_gray_resized = cv2.resize(ir_gray, (w, h))
        ir_img_resized = cv2.resize(ir_img, (w, h))
    else:
        ir_gray_resized = ir_gray
        ir_img_resized = ir_img

    osd_mask_vis = create_osd_mask(h, w)

    # ── 夜间：优先检测配准（特征匹配在夜间不可靠，OSD文字易误匹配）───
    if night_mode:
        osd_mask_ir_orig = create_osd_mask(ir_h, ir_w)
        ir_det  = _detect_target_ir(ir_gray, osd_mask_ir_orig)
        vis_det = _detect_target_vis(vis_gray, osd_mask_vis, night_mode=True)
        if ir_det is not None and vis_det is not None:
            ir_cx, ir_cy = ir_det[0], ir_det[1]
            vis_cx, vis_cy = vis_det[0], vis_det[1]
            # 非均匀缩放：直接用分辨率比（夜间bbox不可靠，不做目标尺寸修正）
            sx = float(w / ir_w)
            sy = float(h / ir_h)
            tx = vis_cx - sx * ir_cx
            ty = vis_cy - sy * ir_cy
            M_det = np.array([[sx, 0, tx], [0, sy, ty]], dtype=np.float64)
            aligned_ir = cv2.warpAffine(ir_img, M_det, (w, h),
                                         borderMode=cv2.BORDER_REFLECT)
            _fill_border(aligned_ir, ir_img_resized, vis_img, None, M_det, h, w)
            return aligned_ir, M_det, None, 0, 0
        return ir_img_resized, None, None, 0, 0

    # ── 白天：特征匹配 + ECC ─────────────────────────────────────
    osd_mask_ir = create_osd_mask(h, w)
    kp_ir, des_ir, _ = extract_edge_features(ir_gray_resized, mask=osd_mask_ir)
    kp_vis, des_vis, _ = extract_edge_features(vis_gray, mask=osd_mask_vis)
    good_matches = match_features(des_ir, des_vis)
    M, mask = estimate_similarity(kp_ir, kp_vis, good_matches)

    n_matches = len(good_matches)
    n_inliers = int(mask.ravel().sum()) if mask is not None else 0

    if M is None:
        # 白天特征匹配失败 → 尝试 ECC 或直接返回缩放
        if use_fine:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            ir_eq = clahe.apply(ir_gray_resized)
            vis_eq = clahe.apply(vis_gray)
            warp = refine_ecc(ir_eq, vis_eq, None, max_iter=500, eps=1e-6)
            aligned_ir = cv2.warpAffine(ir_img_resized, warp, (w, h),
                                         borderMode=cv2.BORDER_REFLECT)
            _fill_border(aligned_ir, ir_img_resized, vis_img, warp, None, h, w)
            return aligned_ir, None, warp, n_matches, n_inliers
        else:
            return ir_img_resized, None, None, n_matches, n_inliers

    # 粗配准：用相似变换 warp IR
    ir_coarse = cv2.warpAffine(ir_img_resized, M, (w, h),
                                borderMode=cv2.BORDER_REFLECT)

    # Fine: ECC 亚像素精配准
    warp = None
    if use_fine:
        ir_coarse_gray = cv2.cvtColor(ir_coarse, cv2.COLOR_BGR2GRAY)
        warp = refine_ecc(ir_coarse_gray, vis_gray, None)
        aligned_ir = cv2.warpAffine(ir_coarse, warp, (w, h),
                                     borderMode=cv2.BORDER_REFLECT)
    else:
        aligned_ir = ir_coarse

    _fill_border(aligned_ir, ir_img_resized, vis_img, warp, M, h, w)

    return aligned_ir, M, warp, n_matches, n_inliers


# ── 基于目标检测的配准 ─────────────────────────────────────────

def _detect_target_ir(gray, osd_mask=None):
    """
    在 IR 图中检测热目标（最亮的连通区域）。
    返回 (cx, cy, bbox_w, bbox_h) 或 None。
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if osd_mask is not None:
        enhanced = cv2.bitwise_and(enhanced, osd_mask)

    # 自适应阈值：取图像最亮像素的 70%
    max_val = enhanced.max()
    thresh_val = int(max_val * 0.7)
    _, binary = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)

    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 选面积最大的连通域作为目标
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 10:
        return None

    M = cv2.moments(c)
    if M['m00'] == 0:
        return None

    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    x, y, bw, bh = cv2.boundingRect(c)
    return cx, cy, bw, bh


def _detect_target_vis(gray, osd_mask=None, night_mode=False):
    """
    在 VIS 图中检测目标。
    - 白天模式（night_mode=False）：检测暗目标（无人机在亮天空中）
    - 夜间模式（night_mode=True） ：检测亮目标（无人机 LED 灯在暗夜空中）
    返回 (cx, cy, bbox_w, bbox_h) 或 None。
    """
    h, w = gray.shape
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 扩大 OSD 安全区：排除顶部 18%、底部 8% 的候选
    margin_top = int(h * 0.18)
    margin_bot = int(h * 0.92)

    if night_mode:
        # ── 夜间：检测亮目标（LED 灯点） ──────────────────────────────
        if osd_mask is not None:
            enhanced = cv2.bitwise_and(enhanced, osd_mask)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        max_val = blurred.max()
        if max_val < 10:
            return None
        thresh_val = int(max_val * 0.5)
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    else:
        # ── 白天：检测暗目标（天空背景-前景）──────────────────────────
        if osd_mask is not None:
            median_val = int(np.median(enhanced[enhanced > 0]))
            enhanced[osd_mask == 0] = median_val
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        bg = cv2.medianBlur(blurred, 51)
        diff = cv2.subtract(bg, blurred)
        max_diff = diff.max()
        if max_diff < 5:
            return None
        thresh_val = int(max_diff * 0.3)
        _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_cx, img_cy = w / 2.0, h / 2.0
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 5:
            continue
        cy_c = y + bh / 2.0
        # 排除 OSD 安全区内的候选
        if cy_c < margin_top or cy_c > margin_bot:
            continue
        cx_c = x + bw / 2.0
        dist_to_center = np.sqrt((cx_c - img_cx)**2 + (cy_c - img_cy)**2)
        candidates.append((c, area, dist_to_center))

    if not candidates:
        return None

    # 跟踪相机中目标通常靠近画面中心，优先选中心附近 + 面积较大的
    # 综合评分：距中心越近越好，面积越大越好
    max_area = max(ca[1] for ca in candidates)
    max_dist = max(ca[2] for ca in candidates) + 1e-6
    best = min(candidates,
               key=lambda ca: ca[2] / max_dist - 0.3 * ca[1] / max_area)

    c = best[0]
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None

    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    x, y, bw, bh = cv2.boundingRect(c)
    return cx, cy, bw, bh


def align_by_detection(ir_path: str, vis_path: str, night_mode: bool = None):
    """
    基于目标检测的跨模态配准。
    分别检测 IR 和 VIS 中的目标，用中心点和尺寸计算仿射变换。

    参数:
        night_mode: None 时自动判断昼夜；True/False 手动指定
    返回:
        aligned_ir, M_affine, ir_det, vis_det
        ir_det/vis_det: (cx, cy, bw, bh) 检测结果
    """
    ir_buf = np.fromfile(ir_path, dtype=np.uint8)
    vis_buf = np.fromfile(vis_path, dtype=np.uint8)
    ir_img = cv2.imdecode(ir_buf, cv2.IMREAD_COLOR)
    vis_img = cv2.imdecode(vis_buf, cv2.IMREAD_COLOR)

    if ir_img is None:
        raise IOError(f"无法读取 IR 图像: {ir_path}")
    if vis_img is None:
        raise IOError(f"无法读取 VIS 图像: {vis_path}")

    ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
    vis_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)

    h_vis, w_vis = vis_gray.shape
    h_ir, w_ir = ir_gray.shape

    if night_mode is None:
        night_mode = is_night_image(vis_gray)

    osd_mask_ir = create_osd_mask(h_ir, w_ir)
    osd_mask_vis = create_osd_mask(h_vis, w_vis)

    ir_det = _detect_target_ir(ir_gray, osd_mask_ir)
    vis_det = _detect_target_vis(vis_gray, osd_mask_vis, night_mode=night_mode)

    if ir_det is None or vis_det is None:
        # 检测失败，返回缩放后的 IR
        ir_resized = cv2.resize(ir_img, (w_vis, h_vis))
        return ir_resized, None, ir_det, vis_det

    ir_cx, ir_cy, ir_bw, ir_bh = ir_det
    vis_cx, vis_cy, vis_bw, vis_bh = vis_det

    # 计算缩放比（IR 坐标系 → VIS 坐标系）
    # 非均匀缩放：分辨率比（夜间 bbox 不可靠时直接用分辨率比）
    sx = float(w_vis / w_ir)
    sy = float(h_vis / h_ir)

    # 白天场景可选用目标尺寸比修正（目标足够大时才可靠）
    if not night_mode:
        ir_diag = np.sqrt(ir_bw**2 + ir_bh**2)
        vis_diag = np.sqrt(vis_bw**2 + vis_bh**2)
        if ir_diag > 5 and vis_diag > 5:
            s_target = vis_diag / ir_diag
            sx *= s_target
            sy *= s_target

    tx = vis_cx - sx * ir_cx
    ty = vis_cy - sy * ir_cy

    M = np.array([
        [sx,  0, tx],
        [0,  sy, ty]
    ], dtype=np.float64)

    aligned_ir = cv2.warpAffine(ir_img, M, (w_vis, h_vis),
                                 borderMode=cv2.BORDER_REFLECT)

    # 边缘填充
    _fill_border(aligned_ir, cv2.resize(ir_img, (w_vis, h_vis)),
                 vis_img, None, M, h_vis, w_vis)

    return aligned_ir, M, ir_det, vis_det


# ── 批量处理 ──────────────────────────────────────────────────

def align_batch(ir_dir: str, vis_dir: str, output_dir: str,
                use_fine: bool = True, callback=None):
    """
    批量配准整个文件夹的 IR/VIS 图像对。

    参数:
        callback: 可选回调函数 callback(idx, total, name, n_matches, n_inliers)
                  用于 GUI 进度更新
    """
    ir_dir = Path(ir_dir)
    vis_dir = Path(vis_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ir_files = {f.stem: f for f in ir_dir.iterdir()
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif')}
    vis_files = {f.stem: f for f in vis_dir.iterdir()
                 if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif')}

    common = sorted(set(ir_files.keys()) & set(vis_files.keys()))
    if not common:
        raise FileNotFoundError(
            f"未找到 IR/VIS 同名配对图像。\nIR: {ir_dir}\nVIS: {vis_dir}")

    results = []
    for idx, name in enumerate(common):
        ir_path = str(ir_files[name])
        vis_path = str(vis_files[name])

        try:
            aligned_ir, H, warp, n_matches, n_inliers = align_single_pair(
                ir_path, vis_path, use_fine=use_fine)

            ext = ir_files[name].suffix
            out_path = str(output_dir / f"{name}{ext}")
            success = cv2.imencode(ext, aligned_ir)[1].tofile(out_path)

            results.append({
                'name': name, 'matches': n_matches,
                'inliers': n_inliers, 'status': 'ok'
            })
        except Exception as e:
            results.append({'name': name, 'matches': 0, 'inliers': 0,
                            'status': str(e)})

        if callback:
            r = results[-1]
            callback(idx + 1, len(common), name,
                     r['matches'], r['inliers'])
        elif (idx + 1) % 10 == 0 or idx == 0:
            skip_tag = " [跳过-已对齐]" if H is None else ""
            print(f"  [{idx+1}/{len(common)}] {name}: "
                  f"{n_matches} matches, {n_inliers} inliers{skip_tag}")

    ok_count = sum(1 for r in results if r['status'] == 'ok')
    print(f"\n配准完成: {ok_count}/{len(common)} 成功")
    avg_matches = np.mean([r['matches'] for r in results if r['status'] == 'ok'])
    print(f"平均匹配点数: {avg_matches:.0f}")

    return results


# ── GUI 界面 ──────────────────────────────────────────────────

def run_gui():
    """启动 Tkinter GUI 界面"""
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    from threading import Thread

    root = tk.Tk()
    root.title("LMAFusion - C2F 跨模态图像配准工具")
    root.geometry("680x520")
    root.resizable(False, False)

    ir_var = tk.StringVar()
    vis_var = tk.StringVar()
    out_var = tk.StringVar()
    fine_var = tk.BooleanVar(value=True)

    # ── 布局 ──
    frame_top = tk.LabelFrame(root, text="文件路径", padx=10, pady=10)
    frame_top.pack(fill='x', padx=10, pady=(10, 5))

    def make_path_row(parent, label, var, row):
        tk.Label(parent, text=label, width=12, anchor='e').grid(
            row=row, column=0, sticky='e', pady=3)
        entry = tk.Entry(parent, textvariable=var, width=45)
        entry.grid(row=row, column=1, padx=5, pady=3)
        def browse():
            d = filedialog.askdirectory()
            if d:
                var.set(d)
        tk.Button(parent, text="浏览...", command=browse, width=8).grid(
            row=row, column=2, pady=3)

    make_path_row(frame_top, "红外图像目录:", ir_var, 0)
    make_path_row(frame_top, "可见光图像目录:", vis_var, 1)
    make_path_row(frame_top, "输出目录:", out_var, 2)

    # 选项
    frame_opt = tk.Frame(root)
    frame_opt.pack(fill='x', padx=10, pady=5)
    tk.Checkbutton(frame_opt, text="启用 ECC 亚像素精配准（推荐）",
                   variable=fine_var).pack(side='left')

    # 进度
    frame_prog = tk.LabelFrame(root, text="配准进度", padx=10, pady=10)
    frame_prog.pack(fill='both', expand=True, padx=10, pady=5)

    progress = ttk.Progressbar(frame_prog, length=600, mode='determinate')
    progress.pack(pady=(0, 5))

    log_text = tk.Text(frame_prog, height=12, width=75, state='disabled',
                       font=('Consolas', 9))
    log_text.pack()

    status_label = tk.Label(root, text="就绪", anchor='w')
    status_label.pack(fill='x', padx=10)

    def log(msg):
        log_text.config(state='normal')
        log_text.insert('end', msg + '\n')
        log_text.see('end')
        log_text.config(state='disabled')

    def on_progress(idx, total, name, matches, inliers):
        progress['maximum'] = total
        progress['value'] = idx
        status_label.config(text=f"[{idx}/{total}] {name}")
        log(f"[{idx}/{total}] {name}: {matches} 匹配, {inliers} 内点")
        root.update_idletasks()

    running = [False]

    def start_align():
        if running[0]:
            return
        ir_d = ir_var.get().strip()
        vis_d = vis_var.get().strip()
        out_d = out_var.get().strip()
        if not ir_d or not vis_d or not out_d:
            messagebox.showwarning("提示", "请先选择所有路径")
            return

        running[0] = True
        log_text.config(state='normal')
        log_text.delete('1.0', 'end')
        log_text.config(state='disabled')
        progress['value'] = 0

        def worker():
            try:
                log(f"IR 目录: {ir_d}")
                log(f"VIS 目录: {vis_d}")
                log(f"输出目录: {out_d}")
                log(f"ECC 精配准: {'开启' if fine_var.get() else '关闭'}")
                log("---")
                results = align_batch(ir_d, vis_d, out_d,
                                      use_fine=fine_var.get(),
                                      callback=on_progress)
                ok = sum(1 for r in results if r['status'] == 'ok')
                log(f"\n--- 配准完成: {ok}/{len(results)} 成功 ---")
                status_label.config(text=f"完成: {ok}/{len(results)} 成功")
                messagebox.showinfo("完成",
                    f"配准完成！\n成功: {ok}/{len(results)}\n输出: {out_d}")
            except Exception as e:
                log(f"\n错误: {e}")
                messagebox.showerror("错误", str(e))
            finally:
                running[0] = False

        Thread(target=worker, daemon=True).start()

    # 按钮
    frame_btn = tk.Frame(root)
    frame_btn.pack(pady=5)
    tk.Button(frame_btn, text="开始配准", command=start_align,
              width=15, height=2, bg='#4CAF50', fg='white',
              font=('Arial', 11, 'bold')).pack(side='left', padx=10)
    tk.Button(frame_btn, text="退出", command=root.quit,
              width=10, height=2).pack(side='left', padx=10)

    root.mainloop()


# ── 主入口 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='C2F 跨模态图像配准工具 (Coarse-to-Fine)')
    parser.add_argument('--ir_dir', type=str, help='红外图像目录')
    parser.add_argument('--vis_dir', type=str, help='可见光图像目录')
    parser.add_argument('--output_dir', type=str, help='对齐后 IR 输出目录')
    parser.add_argument('--no_fine', action='store_true',
                        help='跳过 ECC 精配准，只做粗配准')
    parser.add_argument('--gui', action='store_true', help='启动 GUI 界面')
    args = parser.parse_args()

    if args.gui:
        run_gui()
        return

    if not args.ir_dir or not args.vis_dir or not args.output_dir:
        print("请指定 --ir_dir, --vis_dir, --output_dir，或使用 --gui 启动界面")
        parser.print_help()
        return

    print(f"C2F 跨模态配准")
    print(f"  IR:  {args.ir_dir}")
    print(f"  VIS: {args.vis_dir}")
    print(f"  输出: {args.output_dir}")
    print(f"  ECC 精配准: {'关闭' if args.no_fine else '开启'}")
    print()

    align_batch(args.ir_dir, args.vis_dir, args.output_dir,
                use_fine=not args.no_fine)


if __name__ == '__main__':
    main()
