import cv2
import numpy as np

def add_color_noise(image, mean, stddev):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.int32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

# 读取图像
image = cv2.imread('image.jpg')

# 设置初始噪声参数
mean = 0
stddev = 10

# 创建一个与图像大小相同的全零数组，用于累积噪声
accumulated_noise = np.zeros_like(image, dtype=np.int32)

# 逐步增加噪声
for step in range(40):
    # 添加彩色噪声
    color_noise = np.random.normal(mean, stddev, image.shape).astype(np.int32)
    accumulated_noise = np.clip(accumulated_noise + color_noise, 0, 255)
    noisy_image = np.clip(image + accumulated_noise.astype(np.uint8), 0, 255)

    # 模糊处理
    image = cv2.GaussianBlur(image, (7, 7), 0)

    # 保存图片
    cv2.imwrite(f'noisy_image_step{step}.jpg', noisy_image)

    # 增加噪声参数和模糊程度
    stddev += 5

# 显示原始图像和最终噪声图像
cv2.imshow('Original Image', image)
cv2.imshow('Final Noisy Image', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()