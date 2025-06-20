import torch
import torch.nn as nn
import argparse

#from res_sw import swin_tiny_patch4_window7_224 as create_model
#from model import swin_tiny_patch4_window7_224 as create_model
from sw_bitfit import swin_tiny_patch4_window7_224 as create_model
#from res_sw_ln import swin_tiny_patch4_window7_224 as create_model
#from adapter_sw import swin_tiny_patch4_window7_224 as create_model
#from sw_lora import swin_tiny_patch4_window7_224 as create_model
#from sw_bitfit import swin_tiny_patch4_window7_224 as create_model
#from sw_adapterformer import swin_tiny_patch4_window7_224 as create_model
#from prompt_sw import swin_tiny_patch4_window7_224 as create_model

# 定义一个示例模型，这里以一个简单的卷积神经网络为例


# 创建一个模型实例

model = create_model()

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    #model = create_model(num_classes=args.num_classes).to(device)
    model = create_model()

    #total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Total trainable parameters: {total_params}")

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

        # 统计模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    #pg = [p for p in model.parameters() if p.requires_grad]
    #optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 是否冻结权重
    #parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--freeze-layers', type=bool, default=True)
#可能是不是parameter就没有计算
#或者我直接拿两个相减，就是res-sw所有需要更新的参数减去model所需要的参数就可以得出我需要的参数
    
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    opt = parser.parse_args()

    main(opt)

