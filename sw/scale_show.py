import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# 定义缩放参数
img_size = 384  # 原始尺寸
scale_factor = 10  # 缩放因子，调整为你想要的比例

# 定义缩放的 transforms
data_transform = transforms.Compose([
    transforms.Resize(int(img_size * scale_factor)),  # 缩放
    transforms.ToTensor()
])

# 加载测试图片路径列表（替换为你自己的图片路径）
image_paths = [
    "/mnt/data/lsy/ZZQ/cell.data/EOSINOPHIL/_0_207.jpeg",
    "/mnt/data/lsy/ZZQ/cell.data/LYMPHOCYTE/_0_33.jpeg",
    "/mnt/data/lsy/ZZQ/cell.data/MONOCYTE/_0_147.jpeg",
    "/mnt/data/lsy/ZZQ/cell.data/NEUTROPHIL/_0_208.jpeg"
]

# 定义保存路径
save_path = "/home/lusiyuan/ZZQ/prompt/sw/scale"
os.makedirs(save_path, exist_ok=True)  # 如果文件夹不存在，创建它

# 加载图片并应用变换
transformed_images = []
for img_path in image_paths:
    image = Image.open(img_path).convert("RGB")  # 打开图片并转换为RGB
    transformed_image = data_transform(image)  # 应用缩放操作
    transformed_images.append(transformed_image)

    category = os.path.basename(os.path.dirname(img_path))  # 获取父目录名（类别）
    
    # 保存缩放后的图片，命名为scale_factor_原文件名
    original_filename = os.path.basename(img_path)  # 获取原始文件名
    #new_filename = f"{scale_factor}_{original_filename}"  # 添加缩放因子前缀
    new_filename = f"{scale_factor}_{category}_{original_filename}"  # 添加缩放因子和类别前缀
    save_name = os.path.join(save_path, new_filename)
    
    # 转换为PIL图片并保存
    scaled_image = transforms.ToPILImage()(transformed_image)  # 转换为PIL图片
    scaled_image.save(save_name)

# 可视化原图和缩放后的图片
fig, axes = plt.subplots(2, len(image_paths), figsize=(15, 6))

for i, img_path in enumerate(image_paths):
    # 原图
    original_image = Image.open(img_path).convert("RGB")
    axes[0, i].imshow(original_image)
    axes[0, i].set_title(f"Original Image {i+1}")
    axes[0, i].axis("off")
    
    # 缩放后的图片
    scaled_image = transformed_images[i].permute(1, 2, 0)  # 转换为 HWC 格式以便显示
    axes[1, i].imshow(scaled_image)
    axes[1, i].set_title(f"Scaled Image {i+1}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
