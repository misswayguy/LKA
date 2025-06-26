import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from adpter_vit import vit_base_patch16_224_in21k as create_model
from utils import train_one_epoch, evaluate

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("/your_file_location"):
        os.makedirs("/your_file_location")

    tb_writer = SummaryWriter()

 
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

   
    train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "test"), transform=data_transform["val"])

    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print(f'Using {nw} dataloader workers per process')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             pin_memory=True, num_workers=nw)


    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), f"Weights file: '{args.weights}' not exist."
        weights_dict = torch.load(args.weights, map_location=device)
        
        del_keys = ['head.weight', 'head.bias']
        for k in del_keys:
            weights_dict.pop(k, None)
        print(model.load_state_dict(weights_dict, strict=False))

  
    if args.freeze_layers:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
            else:
                print(f"Training: {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)

    # Cosine learning rate scheduler
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                device=device, epoch=epoch, lr_scheduler=scheduler)

        # Validate
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)

        # Log metrics
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # Save model weights
        torch.save(model.state_dict(),
                   f"/your_file_location/model-epoch-{epoch}.pth")

    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str, default="/your_file_location")
    parser.add_argument('--weights', type=str, default='/your_file_location/vit_base_patch16_224_in21k.pth',
                        help='Path to pretrained weights')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='Device id (e.g., 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
