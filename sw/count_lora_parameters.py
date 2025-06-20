import torch
import torch.nn as nn
from lora_layers_4 import MergedLinear
from lora_sw_4 import swin_tiny_patch4_window7_224  

def count_trainable_parameters(model):
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params

def main():
    model = swin_tiny_patch4_window7_224(num_classes=1000)

    # 只训练 MergedLinear 层
    for name, param in model.named_parameters():
        if "MergedLinear" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 统计可训练参数量
    trainable_params = count_trainable_parameters(model)
    print(f"Total trainable parameters in MergedLinear: {trainable_params}")

if __name__ == "__main__":
    main()
