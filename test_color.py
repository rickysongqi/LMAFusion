import cv2
import numpy as np

# load the black and white fusion
bw_path = './results_MSRS/00004N.png'
bw = np.fromfile(bw_path, dtype=np.uint8)
bw = cv2.imdecode(bw, cv2.IMREAD_GRAYSCALE)

# load the color vis
vis_path = 'C:/Users/Ricky-Li/Desktop/毕业设计/MSRS-main/test/vi/00004N.png'
vis = np.fromfile(vis_path, dtype=np.uint8)
vis = cv2.imdecode(vis, cv2.IMREAD_COLOR)

# resize color to bw if needed
h, w = bw.shape
vis_resized = cv2.resize(vis, (w, h))

# BGR to YCrCb
vis_ycrcb = cv2.cvtColor(vis_resized, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(vis_ycrcb)

# merge and convert back
fused_ycrcb = cv2.merge([bw, cr, cb])
final_color = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)

# save
success = cv2.imwrite('test_color_output.png', final_color)
print(f"Colorization test success: {success}, shape: {final_color.shape}")
