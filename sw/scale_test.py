import os
import json
import argparse
import sys

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image  # 添加导入
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from utils import read_split_data
from my_dataset import MyDataSet
from model import swin_large_patch4_window7_224_in22k as create_model


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def save_scaled_images(image_paths, image_labels, transform, output_dir, scale_factor):
    """
    保存缩放后的图片到指定目录，同时按照类别创建子文件夹
    :param image_paths: 图片路径列表
    :param image_labels: 图片对应的类别列表
    :param transform: 缩放的transforms
    :param output_dir: 图片保存根目录
    :param scale_factor: 缩放因子
    """
    os.makedirs(output_dir, exist_ok=True)
    for label in set(image_labels):
        os.makedirs(os.path.join(output_dir, f"{scale_factor}_{label}"), exist_ok=True)

    for img_path, label in zip(image_paths, image_labels):
        # 加载原始图片
        image = Image.open(img_path).convert("RGB")
        # 应用缩放变换
        scaled_image = transform(image)
        # 反归一化
        scaled_image = scaled_image.permute(1, 2, 0).numpy()  # 转为HWC格式
        scaled_image = (scaled_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])  # 反归一化
        scaled_image = (scaled_image * 255).clip(0, 255).astype(np.uint8)  # 转为uint8

        # 保存图片
        img_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f"{scale_factor}_{label}", f"{scale_factor}_{label}_{img_name}")
        Image.fromarray(scaled_image).save(save_path)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    _, _, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 384
    scale_factor = 1.143  # 设置缩放因子
    data_transform = transforms.Compose([
        transforms.Resize(int(img_size * scale_factor)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 保存缩放后的图片
    output_dir = "/home/lusiyuan/ZZQ/prompt/sw/scale"
    save_scaled_images(val_images_path, val_images_label, data_transform, output_dir, scale_factor)

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes)
    # load pretrain weights
    assert os.path.exists(args.weights), "cannot find {} file".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    model.eval()
    with torch.no_grad():
        for val_data in tqdm(val_loader, file=sys.stdout):
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--data-path', type=str, default="/mnt/data/lsy/ZZQ/cell.data")
    parser.add_argument('--weights', type=str, default='/mnt/data/lsy/ZZQ/weights_sw_large_cell_best.pth',
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
