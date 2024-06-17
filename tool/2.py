import cv2
import numpy as np

# 读取图像
image = cv2.imread('image.jpg')

# 生成纯三色噪声图像
noise_image = np.random.randint(0, 256, image.shape, dtype=np.uint8)

# 保存纯三色噪声图像
cv2.imwrite('noise_image.jpg', noise_image)

# 显示纯三色噪声图像
cv2.imshow('Noise Image', noise_image)
cv2.waitKey(0)
cv2.destroyAllWindows()