"""
VIS/IR 视场对齐诊断工具
=======================
用法：
  python align_check.py

功能：
  1. 并排显示 IR 和 VIS 原图（resize 前），直接看位置偏差
  2. 显示当前 dataset.py 输出的 IR/VIS patch 的叠加图（绿=IR边缘，红=VIS边缘）
  3. 帮助你目视确定 VIS_CROP_* 参数应该设置为多少

交互：
  按 'q' 或 ESC 退出
"""

import cv2
import numpy as np
from pathlib import Path

# ── 配置 ────────────────────────────────────────────────────
DATA_DIR   = './data_clean/val'
IR_DIR     = DATA_DIR + '/ir'
VIS_DIR    = DATA_DIR + '/vis'
NUM_SHOW   = 6  # 显示前 N 对图像

# VIS 裁剪参数（先全图，设为 None = 不裁剪）
# 目视确认后填入正确值；格式：(x1, y1, x2, y2)
VIS_CROP   = None  # 例如 (300, 0, 1100, 576)
# ────────────────────────────────────────────────────────────


def sobel(img):
    g = cv2.Sobel(img, cv2.CV_32F, 1, 1, ksize=3)
    return cv2.convertScaleAbs(g)


def overlay_edges(ir, vis, alpha=0.6):
    """
    将 IR 边缘（绿）和 VIS 边缘（红）叠加在一张图上。
    完全对齐时两组边缘应重合（变黄/橙）。
    有偏差时会看到两组分开的彩色轮廓。
    """
    h, w = ir.shape[:2]
    vis_rs = cv2.resize(vis, (w, h))

    edge_ir  = sobel(ir)
    edge_vis = sobel(vis_rs)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :, 1] = edge_ir    # 绿通道 = IR 边缘
    canvas[:, :, 2] = edge_vis   # 红通道 = VIS 边缘

    base = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 1 - alpha, canvas, alpha, 0)


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def main():
    ir_files  = sorted(Path(IR_DIR).glob('*.png'))[:NUM_SHOW]
    vis_files = [Path(VIS_DIR) / f.name for f in ir_files]

    for ir_path, vis_path in zip(ir_files, vis_files):
        if not vis_path.exists():
            print(f"缺失 VIS: {vis_path.name}")
            continue

        ir_orig  = load_gray(ir_path)
        vis_orig = load_gray(vis_path)
        ir_h, ir_w = ir_orig.shape
        vis_h, vis_w = vis_orig.shape

        # ── 面板 A：并排原图（不做任何操作）──
        ir_disp = cv2.resize(ir_orig, (640, 480))
        vis_disp = cv2.resize(vis_orig, (640, 480))
        panel_raw = np.concatenate([
            cv2.cvtColor(ir_disp,  cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(vis_disp, cv2.COLOR_GRAY2BGR),
        ], axis=1)
        cv2.putText(panel_raw, f'IR {ir_w}x{ir_h}',  (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(panel_raw, f'VIS {vis_w}x{vis_h}', (660, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

        # ── 面板 B：视场对齐后的叠加边缘图 ──
        if VIS_CROP is not None:
            x1, y1, x2, y2 = VIS_CROP
            vis_cropped = vis_orig[y1:y2, x1:x2]
        else:
            vis_cropped = vis_orig

        overlay = overlay_edges(ir_orig, vis_cropped)
        overlay_disp = cv2.resize(overlay, (1280, 480))
        cv2.putText(overlay_disp,
                    'Edge Overlay: GREEN=IR, RED=VIS (aligned = yellow/orange)',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(overlay_disp,
                    f'VIS_CROP={VIS_CROP}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # ── 合并展示 ──
        combined = np.concatenate([panel_raw, overlay_disp], axis=0)
        cv2.imshow(f'Alignment Check: {ir_path.name}', combined)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key in [ord('q'), 27]:
            break

    print("诊断完成。根据叠加图確認 VIS_CROP 参数后，")
    print("填入 dataset.py 的 VIS_CROP_X1/Y1/X2/Y2 变量。")


if __name__ == '__main__':
    main()
