import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from three import convnext_small as create_model


from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("/mnt/data/lsy/ZZQ/weights_convnext_three_brain") is False:
        os.makedirs("/mnt/data/lsy/ZZQ/weights_convnext_three_brain")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

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

    #total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Total trainable parameters: {total_params}")

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

    #pg = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5E-2)


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

        #tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tags = ["train_loss", "train_acc", "val_loss", "val_acc"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        #tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "/mnt/data/lsy/ZZQ/weights_convnext_three_brain/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    #parser.add_argument('--data-path', type=str,
     #                   default="/data/flower_photos")

    #parser.add_argument('--data-path', type=str,
    #                  default="/home/z/zz257/project/prompt/promot/cell.data")

    #parser.add_argument('--data-path', type=str,
    #                    default="/home/z/zz257/project/prompt/promot/Covid19-dataset")
    
    #parser.add_argument('--data-path', type=str,
    #                    default="/home/z/zz257/project/prompt/promot/covid_19_Ra")

    parser.add_argument('--data-path', type=str,
                       default="/mnt/data/lsy/ZZQ/Brain_tumor_harvest")

    #parser.add_argument('--data-path', type=str,
    #                 default="/home/z/zz257/project/prompt/promot/Breast_Ultr_sound_Images_Dataset")
    
    #parser.add_argument('--data-path', type=str,
    #                   default="/mnt/data/lsy/ZZQ/TB_Chest_Radiography_Database")

    #parser.add_argument('--data-path', type=str,
    #                   default="/home/z/zz257/project/prompt/promot/SoybeanSeeds")
    

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/home/lusiyuan/ZZQ/prompt/convnext/convnext_small_1k_224_ema.pth',
                        help='initial weights path')
    
    #parser.add_argument('--weights', type=str, default='/home/z/zz257/project/prompt/promot/sw/swin_base_patch4_window12_384_22k.pth',
    #                    help='initial weights path')

    #parser.add_argument('--weights', type=str, default='/home/z/zz257/project/prompt/promot/sw/swin_large_patch4_window7_224_22k.pth',
    #                    help='initial weights path')

    # 是否冻结权重
    #parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1,2,3,4,5 or cpu)') #改GPU

    opt = parser.parse_args()

    main(opt)