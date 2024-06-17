import os
import random
from PIL import Image

def random_crop_image(source_dir, destination_dir, crop_size, num_crops):
    index = 0
    # 遍历源目录中的子目录
    for root, dirs, files in os.walk(source_dir):
        print(len(files))
        for dir_name in dirs:
            print(dir_name)
            source_subdir = os.path.join(root, dir_name)
            destination_subdir = os.path.join(destination_dir, dir_name)

            print(source_subdir)
            print(destination_subdir)

            # 创建目标子目录
            os.makedirs(destination_subdir, exist_ok=True)

            # 遍历子目录中的图片文件
            for file_name in os.listdir(source_subdir):
                source_file = os.path.join(source_subdir, file_name)
                target_folder_name = file_name[:7].replace(" ", "-")
                #destination_file_prefix = os.path.join(destination_subdir, os.path.splitext(file_name)[0])



                # 打开原始图片
                image = Image.open(source_file)

                # 剪裁并保存多个随机剪裁后的图片
                for i in range(num_crops):
                    # 随机生成剪裁区域的左上角坐标
                    left = random.randint(0, image.width - crop_size)
                    top = random.randint(0, image.height - crop_size)

                    # 剪裁图片
                    cropped_image = image.crop((left, top, left + crop_size, top + crop_size))

                    index_str = str(index).zfill(6)
                    # 生成目标文件路径并保存剪裁后的图片
                    destination_file = f"{target_folder_name}_{index_str}.jpg"
                    destination_file = os.path.join(destination_subdir, destination_file)
                    cropped_image.save(destination_file)
                    index +=1

                # 关闭原始图片
                image.close()

# 源目录
source_directory = "G:/DATA/SatelliteImage/data"
# 目标目录
destination_directory = "G:/DATA/JinShu/clip"

# 剪裁尺寸
crop_size = 512
# 每张图片剪裁数量
num_crops_per_image = 5

# 调用函数进行随机剪裁
random_crop_image(source_directory, destination_directory, crop_size, num_crops_per_image)