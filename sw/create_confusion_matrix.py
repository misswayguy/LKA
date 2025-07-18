import os
import json
import argparse
import sys

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from utils import read_split_data
from my_dataset import MyDataSet
#from model import swin_tiny_patch4_window7_224 as create_model
from model import swin_base_patch4_window12_384_in22k as create_model
#from res_sw import swin_tiny_patch4_window7_224 as create_model
#from res_sw_ln import swin_tiny_patch4_window7_224 as create_model
#from res_sw_pro import swin_tiny_patch4_window7_224 as create_model
#from res_sw_inside import swin_tiny_patch4_window7_224 as create_model
#from out_seven import swin_tiny_patch4_window7_224 as create_model
#from average_duo import swin_tiny_patch4_window7_224 as create_model
#from sansan_con import swin_tiny_patch4_window7_224 as create_model
#from seven_cov import swin_base_patch4_window12_384 as create_model
#from dilute_seven import swin_tiny_patch4_window7_224 as create_model
#from seven_cov import swin_large_patch4_window7_224_in22k as create_model
#from dw_seven import swin_tiny_patch4_window7_224 as create_model
#from dw_five import swin_base_patch4_window12_384_in22k as create_model
#from dw_three import swin_base_patch4_window12_384_in22k as create_model
#from adapter import swin_base_patch4_window12_384_in22k as create_model
#from seven_cov import swin_large_patch4_window7_224_in22k as create_model
#from nine_conv import swin_tiny_patch4_window7_224 as create_model
#from concat_duo import swin_tiny_patch4_window7_224 as create_model
#from sum_duo import swin_tiny_patch4_window7_224 as create_model
#from max_duo import swin_tiny_patch4_window7_224 as create_model
#from eleven_conv import swin_tiny_patch4_window7_224 as create_model
#from weight_duo import swin_tiny_patch4_window7_224 as create_model
#from conv_byp import swin_tiny_patch4_window7_224 as create_model
#from ciat import swin_tiny_patch4_window7_224 as create_model
#from st_adapter import swin_tiny_patch4_window7_224 as create_model
#from aim import swin_tiny_patch4_window7_224 as create_model
#from seven_cov import swin_tiny_patch4_window7_224 as create_model
#from seven_ln_cov import swin_tiny_patch4_window7_224 as create_model
#from five_cov import swin_tiny_patch4_window7_224 as create_model
#from RepAdapter import swin_tiny_patch4_window7_224 as create_model
#from res_sw import swin_base_patch4_window12_384_in22k as create_model
#from res_sw import swin_large_patch4_window7_224_in22k as create_model
#from v2_res_sw import swin_tiny_patch4_window7_224 as create_model
#from prompt_sw import swin_tiny_patch4_window7_224 as create_model
#from adapter_sw import swin_tiny_patch4_window7_224 as create_model
#from v2_RVFL_model import swin_tiny_patch4_window7_224 as create_model
#from sw_lora import swin_tiny_patch4_window7_224 as create_model
#from sw_bitfit import swin_tiny_patch4_window7_224 as create_model
#from sw_adapterformer import swin_tiny_patch4_window7_224 as create_model
#from v3_res_sw_msa import swin_tiny_patch4_window7_224 as create_model
#from adapter import swin_base_patch4_window12_384_in22k as create_model
#from dw_five import swin_large_patch4_window7_224_in22k as create_model
#from dw_three import swin_tiny_patch4_window7_224 as create_model
#from adapter import swin_tiny_patch4_window7_224 as create_model
#from dw_five import swin_tiny_patch4_window7_224 as create_model
#from dw_seven import swin_tiny_patch4_window7_224 as create_model
#from lora_sw import swin_tiny_patch4_window7_224 as create_model
#from lora_sw_4 import swin_tiny_patch4_window7_224 as create_model
#from lora_sw_8 import swin_tiny_patch4_window7_224 as create_model
#from lora_sw_16 import swin_tiny_patch4_window7_224 as create_model
#from dw_seven import swin_large_patch4_window7_224_in22k as create_model
#from model import swin_large_patch4_window7_224_in22k as create_model
#from adapter import swin_large_patch4_window7_224_in22k as create_model
#from adapter import swin_tiny_patch4_window7_224 as create_model


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


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    _, _, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 384
    data_transform = {
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

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
    #model.load_state_dict(torch.load(args.weights, map_location=device), strict=True) #prompt
    model.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)
    model.eval()
    with torch.no_grad():
        for val_data in tqdm(val_loader, file=sys.stdout):
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)

     # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    #parser.add_argument('--data-path', type=str,
     #                   default="/data/flower_photos")

    #parser.add_argument('--data-path', type=str,
    #              default="/mnt/data/lsy/ZZQ/cell.data")

    parser.add_argument('--data-path', type=str,
                        default="/mnt/data/lsy/ZZQ/covid")
    
    #parser.add_argument('--data-path', type=str,
    #                    default="/home/z/zz257/project/prompt/promot/covid_19_Ra")

    #parser.add_argument('--data-path', type=str,
    #                   default="/mnt/data/lsy/ZZQ/Brain_tumor_harvest")

    #parser.add_argument('--data-path', type=str,
    #                 default="/mnt/data/lsy/ZZQ/Breast_Ultr_sound_Images_Dataset")
    
    #parser.add_argument('--data-path', type=str,
    #                   default="/mnt/data/lsy/ZZQ/TB_Chest_Radiography_Database")
    
    # 训练权重路径
    parser.add_argument('--weights', type=str, default='/mnt/data/lsy/ZZQ/weights_base_covid_2_best',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
