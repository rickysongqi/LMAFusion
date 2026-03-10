"""
精确标定：在图像上点击，记录无人机中心坐标
用于计算正确的 VIS_CROP 参数
"""
import cv2
import numpy as np

clicked = []

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))
        print(f"点击: ({x}, {y})")
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('click', param)

def mark_drone(img_path, label):
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取: {img_path}")
        return None
    h, w = img.shape[:2]
    # 缩放到最大 1280 宽用于显示
    scale = min(1280 / w, 800 / h)
    disp = cv2.resize(img, (int(w*scale), int(h*scale)))
    print(f"\n{label}: {w}x{h}")
    print("请点击无人机中心，然后按任意键")
    clicked.clear()
    cv2.imshow('click', disp)
    cv2.setMouseCallback('click', mouse_cb, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if clicked:
        # 还原到原图坐标
        rx = int(clicked[-1][0] / scale)
        ry = int(clicked[-1][1] / scale)
        print(f"原图坐标: ({rx}, {ry})")
        return (rx, ry), (w, h)
    return None

# === 标定图像（选一帧有无人机清晰可见的）===
SAMPLE = '20190925_101846_1_100075.png'
IR_PATH  = f'./data_clean/val/ir/{SAMPLE}'
VIS_PATH = f'./data_clean/val/vis/{SAMPLE}'

print("=== 第1步：点击 IR 图像中无人机中心 ===")
result_ir = mark_drone(IR_PATH,  'IR')
print("\n=== 第2步：点击 VIS 图像中无人机中心 ===")
result_vis = mark_drone(VIS_PATH, 'VIS')

if result_ir and result_vis:
    (ir_x, ir_y), (IR_W, IR_H)   = result_ir
    (vis_x, vis_y), (VIS_W, VIS_H) = result_vis

    print(f"\n=== 标定结果 ===")
    print(f"IR  ({IR_W}x{IR_H}): 无人机在 ({ir_x}, {ir_y})")
    print(f"VIS ({VIS_W}x{VIS_H}): 无人机在 ({vis_x}, {vis_y})")

    # 计算最优裁剪
    norm_x = ir_x / IR_W
    norm_y = ir_y / IR_H
    aspect = IR_W / IR_H  # 640/369

    max_ch_top = vis_y / norm_y
    max_ch_bot = (VIS_H - vis_y) / (1 - norm_y)
    crop_h = int(min(max_ch_top, max_ch_bot))
    crop_w = int(crop_h * aspect)

    y1 = max(0, int(vis_y - norm_y * crop_h))
    y2 = min(VIS_H, y1 + crop_h)
    x1 = max(0, int(vis_x - norm_x * crop_w))
    x2 = min(VIS_W, x1 + crop_w)

    print(f"\n=== 推荐 VIS_CROP ===")
    print(f"VIS_CROP = ({x1}, {y1}, {x2}, {y2})")
    print(f"裁剪尺寸: {x2-x1}x{y2-y1}")
