import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import sys

from sklearn.metrics import average_precision_score, precision_recall_curve

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model.MLDataset import MultiLabelDataset
from model.QST import SwinQ2L

def paser_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="", help="path to the dataset folder, e.g. /root/plant_dataset/val/")
    parser.add_argument('--weights', type=str, default='', help='path to the weights file, e.g. /root/weights.pth')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    
    return opt

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []  # 预测结果
    targets = []  # 真实标签
    data_loader = tqdm(data_loader, file=sys.stdout)  # 进度条
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions.extend(outputs.view(-1).cpu().detach().numpy())
        targets.extend(labels.view(-1).cpu().detach().numpy())
    return predictions, targets

def main():
    opt = paser_options()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    data_transform = transforms.Compose([transforms.Resize(int(448 * 1.143)),
                                         transforms.CenterCrop(448),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    test_dataset = MultiLabelDataset(opt.data_path, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn)

    model = SwinQ2L(num_classes=6)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint["model"])

    predictions, targets = evaluate(model, test_loader, device)
    predictions = np.array(predictions)
    targets = np.array(targets)
    print(predictions.shape)
    print(targets.shape)
    print(average_precision_score(targets, predictions, average='micro'))
    print(average_precision_score(targets, predictions, average='macro'))

    # 画PR曲线
    precision, recall, _ = precision_recall_curve(targets.ravel(), predictions.ravel())
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.show()

if __name__ == '__main__':
    main()