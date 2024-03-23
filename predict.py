import os
import argparse
from PIL import Image

import torch
from torchvision import transforms

from model.QST import SwinQ2L

num_classes = 6
imsize = 224
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-path', type=str, default="", help='Your path of single plant pathology image.')
    parser.add_argument('--weights', type=str, default="", help='Path of model weights, which ends with "pth".')

    args = parser.parse_args()
    return args

def main():
    args = args_parse()
    data_path = os.path.join(args.img_path)
    weights = os.path.join(args.weights)

    print("Using {} device training.".format(device.type))

    # 实例化数据集
    data_transform = {
        "test": transforms.Compose([transforms.Resize(int(imsize * 1.143)),
                                    transforms.CenterCrop(imsize),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    img = data_transform["test"](Image.open(data_path))

    # 实例化模型
    model = SwinQ2L(num_classes=num_classes)
    model.to(device)

    # 加载预训练权重
    if weights != '':
        assert os.path.exists(weights), "weights file: '{}' not exist.".format(weights)
        weights_dict = torch.load(weights, map_location=device)["model"]
        print(model.load_state_dict(weights_dict, strict=False))

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(img.unsqueeze(0).to(device))
        print(outputs)
        print(torch.sigmoid(outputs))

if __name__ == '__main__':
    main()