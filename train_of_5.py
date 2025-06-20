import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import swin_base_patch4_window12_384_in22k as create_model
from utils import train_one_epoch, evaluate

import pickle
import random

def save_fixed_test_set(root: str, test_ratio: float = 0.2, save_path: str = "fixed_test_set.pkl"):
    """
    固定测试集并保存到文件。
    """
    assert os.path.exists(root), f"Dataset root: {root} does not exist."

    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    classes.sort()
    supported = [".gif", ".GIF", ".jpeg", ".JPEG", ".jpg", ".JPG", ".png", ".PNG"]

    test_images = []
    test_labels = []
    class_indices = {cla: idx for idx, cla in enumerate(classes)}

    for cla in classes:
        class_path = os.path.join(root, cla)
        images = [os.path.join(class_path, img) for img in os.listdir(class_path)
                  if os.path.splitext(img)[-1] in supported]
        images.sort()
        test_count = int(len(images) * test_ratio)
        test_samples = random.sample(images, test_count)

        test_images.extend(test_samples)
        test_labels.extend([class_indices[cla]] * len(test_samples))

    with open(save_path, 'wb') as f:
        pickle.dump((test_images, test_labels), f)

    print(f"Fixed test set saved to {save_path}.")

def load_fixed_test_set(save_path: str):
    """
    加载固定测试集。
    """
    assert os.path.exists(save_path), f"Test set file {save_path} does not exist."
    with open(save_path, 'rb') as f:
        test_images, test_labels = pickle.load(f)
    return test_images, test_labels

def create_train_set(root: str, train_ratio: float, fixed_test_path: str):
    """
    根据指定训练集比例创建训练集，测试集保持不变。
    """
    assert os.path.exists(root), f"Dataset root: {root} does not exist."

    test_images, _ = load_fixed_test_set(fixed_test_path)

    classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    classes.sort()
    supported = [".gif", ".GIF", ".jpeg", ".JPEG", ".jpg", ".JPG", ".png", ".PNG"]

    train_images = []
    train_labels = []
    class_indices = {cla: idx for idx, cla in enumerate(classes)}

    for cla in classes:
        class_path = os.path.join(root, cla)
        images = [os.path.join(class_path, img) for img in os.listdir(class_path)
                  if os.path.splitext(img)[-1] in supported]
        images.sort()

        train_candidates = [img for img in images if img not in test_images]
        train_count = int(len(train_candidates) * train_ratio)
        train_samples = train_candidates[:train_count]

        train_images.extend(train_samples)
        train_labels.extend([class_indices[cla]] * len(train_samples))

    return train_images, train_labels

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("/mnt/data/lsy/ZZQ/weights_base_covid_5") is False:
        os.makedirs("/mnt/data/lsy/ZZQ/weights_base_covid_5")

    tb_writer = SummaryWriter()

    fixed_test_path = "fixed_test_set.pkl"

    # 如果测试集尚未生成，先固定生成 20% 测试集
    if not os.path.exists(fixed_test_path):
        save_fixed_test_set(args.data_path, test_ratio=0.2, save_path=fixed_test_path)

    # 创建训练集（动态调整训练比例）
    train_images_path, train_images_label = create_train_set(args.data_path, train_ratio=args.train_ratio, fixed_test_path=fixed_test_path)

    # 加载固定测试集
    val_images_path, val_images_label = load_fixed_test_set(fixed_test_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5E-2)

    acc_best = 0

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)

        if val_acc > acc_best:
            torch.save(model.state_dict(), "/mnt/data/lsy/ZZQ/weights_base_covid_0.05_best.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str, default="/mnt/data/lsy/ZZQ/covid")
    parser.add_argument('--weights', type=str, default='/home/lusiyuan/ZZQ/prompt/sw/swin_base_patch4_window12_384_22k.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:5', help='device id (i.e. 5 or 0,1,2,3,4,5 or cpu)')
    parser.add_argument('--train-ratio', type=float, default=0.05, help='training set ratio (e.g., 0.05 for 5%)')

    opt = parser.parse_args()

    model = create_model()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    main(opt)
