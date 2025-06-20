import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
# from model import swin_tiny_patch4_window7_224 as create_model
# from model import swin_large_patch4_window7_224_in22k as create_model
# from model import swin_base_patch4_window12_384_in22k as create_model
# from dw_fiftyone_five import swin_tiny_patch4_window7_224 as create_model
# from adapter import swin_tiny_patch4_window7_224 as create_model
#from dw_thirtyone import swin_tiny_patch4_window7_224 as create_model
# from dw_one import swin_tiny_patch4_window7_224 as create_model
# from dw_one import swin_large_patch4_window7_224_in22k as create_model
#from dw_one import swin_base_patch4_window12_384_in22k as create_model
from adapterformer import swin_base_patch4_window12_384_in22k as create_model
# from sw_lora import swin_base_patch4_window12_384_in22k as create_model

from utils import train_one_epoch, evaluate

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # if os.path.exists("fiftyone_five_weights_sw_tiny_covid_1") is False:
    #     os.makedirs("fiftyone_five_weights_sw_tiny_covid_1")

    tb_writer = SummaryWriter()

    # 直接使用已经分好的 train 和 test 数据集路径
    train_images_dir = os.path.join(args.data_path, "train")
    val_images_dir = os.path.join(args.data_path, "test")
    # val_images_dir = os.path.join(args.data_path, "val") 
    

    # 获取 train 和 test 文件夹中的所有图像文件路径
    def get_image_paths(root_dir):
        image_paths = []
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path):  # 确保是文件而不是目录
                        image_paths.append(img_path)
        return image_paths

    # def get_image_paths(root_dir): #tiny imagenet 
    #     image_paths = []
    #     for class_name in os.listdir(root_dir):
    #         class_dir = os.path.join(root_dir, class_name)
    #         images_subdir = os.path.join(class_dir, "images")  # 👈 多加一层
    #         if os.path.isdir(images_subdir):
    #             for img_name in os.listdir(images_subdir):
    #                 if img_name.endswith((".jpg", ".jpeg", ".png", ".JPEG", ".JPG")):
    #                     img_path = os.path.join(images_subdir, img_name)
    #                     image_paths.append(img_path)
    #     return image_paths


    train_images_path = get_image_paths(train_images_dir)
    val_images_path = get_image_paths(val_images_dir)

    # def get_labels(image_paths, root_dir): #tiny imagenet 
    #     labels = []
    #     class_names = sorted(os.listdir(root_dir))  # 保证类名和下标一一对应
    #     class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    #     for img_path in image_paths:
    #     # 从路径中提取类别名（即两级目录上）
    #         class_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
    #         if class_name not in class_to_idx:
    #             raise ValueError(f"Unexpected class name '{class_name}' in path '{img_path}'")
    #         labels.append(class_to_idx[class_name])
    #     return labels


    # 生成标签（假设文件夹名称为类别名）
    def get_labels(image_paths, root_dir):
        labels = []
        class_names = os.listdir(root_dir)
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        for img_path in image_paths:
            class_name = os.path.basename(os.path.dirname(img_path))
            labels.append(class_to_idx[class_name])
        return labels

    train_images_label = get_labels(train_images_path, train_images_dir)
    val_images_label = get_labels(val_images_path, val_images_dir)

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

    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除head外，其他权重全部冻结
    #         if "head" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5E-2)

    if args.freeze_layers:
    # 打印冻结信息
        print("="*30 + " Freezing Strategy " + "="*30)
    
        for name, param in model.named_parameters():
        # 更精确的head层判断
            if "head" in name.lower():  # 兼容不同命名习惯（Head/head/classifier等）
                param.requires_grad = True
                print(f"Training layer: {name}")
            else:
                param.requires_grad = False
    
    # 验证冻结效果
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params}/{total_params} ({100.*trainable_params/total_params:.2f}%)")
        print("="*75)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5E-2)

    acc_best = 0

    for epoch in range(args.epochs):
        # train
        # train_loss, train_acc = train_one_epoch(model=model,
        #                                         optimizer=optimizer,
        #                                         data_loader=train_loader,
        #                                         device=device,
        #                                         epoch=epoch)

        # # validate
        # val_loss, val_acc = evaluate(model=model,
        #                              data_loader=val_loader,
        #                              device=device,
        #                              epoch=epoch)
        
        

        # tags = ["train_loss", "train_acc", "val_loss", "val_acc"]
        # tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)

        train_loss, train_acc, train_f1, train_sen = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        
        val_loss, val_acc, val_f1, val_sen = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "train_f1", "train_sensitivity", "val_loss", "val_acc", "val_f1", "val_sensitivity"]

        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], train_f1, epoch)
        tb_writer.add_scalar(tags[3], train_sen, epoch)
        tb_writer.add_scalar(tags[4], val_loss, epoch)
        tb_writer.add_scalar(tags[5], val_acc, epoch)
        tb_writer.add_scalar(tags[6], val_f1, epoch)
        tb_writer.add_scalar(tags[7], val_sen, epoch)


        # if val_acc > acc_best:
        #     torch.save(model.state_dict(), "/mnt/data/lsy/ZZQ/fiftyone_five_weights_sw_tiny_covid_1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str,
                      default="/mnt/data/lsy/ZZQ/tiny_imagenet_crosee_validation/train_0/fold_4")
    # parser.add_argument('--weights', type=str, default='/home/lusiyuan/ZZQ/prompt/sw/swin_tiny_patch4_window7_224.pth',
    #                      help='initial weights path')
    parser.add_argument('--weights', type=str, default='/home/lusiyuan/ZZQ/prompt/sw/swin_base_patch4_window12_384_22k.pth',
                         help='initial weights path')
    # parser.add_argument('--freeze-layers', type=bool, default=True)
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--freeze-layers', action='store_true',
                   help='Freeze all layers except head')  # ✅ 正确写法
    parser.add_argument('--device', default='cuda:5', help='device id (i.e. 0 or 0,1,2,3,4,5 or cpu)')

    opt = parser.parse_args()

    model = create_model()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    main(opt)