import os
import argparse
import sys
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# from pytorch_metric_learning import losses

from sklearn.metrics import average_precision_score

from model.MLDataset import MultiLabelDataset
from model.QST import SwinQ2L
from model.simple_asymmetric_loss import AsymmetricLoss

def parser_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--imsize', type=int, default=384)
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate of adamw optimizer')
    parser.add_argument('--max-lr', type=float, default=0.1,
                        help='max learning rate of one-cycle-lr scheduler')
    parser.add_argument('--weight-decay', type=float, default=0.4,
                        help='weight decay of adamw optimizer')

    parser.add_argument('--data-path', type=str,
                        default="", help="path to the WHOLE dataset folder, e.g. /root/plant_dataset/")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path, set empty to train from scratch')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    return opt

def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion):
    model.train()
    optimizer.zero_grad()

    predictions = []  # 预测结果
    targets = []  # 真实标签
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    data_loader = tqdm(data_loader, file=sys.stdout)  # 进度条
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels) * 100
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        accu_loss += loss.item()

        predictions.extend(outputs.view(-1).cpu().detach().numpy())
        targets.extend(labels.view(-1).cpu().detach().numpy())
        mAP = average_precision_score(targets, predictions)

        data_loader.set_description("[Train epoch {}] loss: {:.3f}, mAP: {:.3f}"
                                    .format(epoch, accu_loss.item() / (step + 1), mAP))
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), mAP

def evaluate(model, data_loader, device, epoch, criterion):
    model.eval()

    predictions = []
    targets = []
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    data_loader = tqdm(data_loader, file=sys.stdout)  # 进度条
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels) * 100
        
        accu_loss += loss.item()

        predictions.extend(outputs.view(-1).cpu().detach().numpy())
        targets.extend(labels.view(-1).cpu().detach().numpy())
        mAP = average_precision_score(targets, predictions)

        data_loader.set_description("[Validate epoch {}] loss: {:.3f}, mAP: {:.3f}"
                                    .format(epoch, accu_loss.item() / (step + 1), mAP))

    return accu_loss.item() / (step + 1), mAP

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    img_size = args.imsize
    batch_size = args.batch_size

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    data_transform = {
        "train": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                     transforms.CenterCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # data_transform = {
    #     "train1": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
    #                                  transforms.CenterCrop(img_size),
    #                                  transforms.ToTensor(),
    #                                  transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #     "train2": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
    #                                  transforms.CenterCrop(img_size),
    #                                  transforms.RandomHorizontalFlip(p=1),
    #                                  transforms.ToTensor(),
    #                                  transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    #     "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
    #                                transforms.CenterCrop(img_size),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    # 实例化数据集
    train_dataset = MultiLabelDataset(os.path.join(args.data_path, 'train'), transform=data_transform["train"])
    # train2_dataset = MultiLabelDataset(args.data_path + 'train/', transform=data_transform["train2"])
    # train_dataset = torch.utils.data.ConcatDataset([train1_dataset, train2_dataset])
    val_dataset = MultiLabelDataset(os.path.join(args.data_path, 'test'), transform=data_transform["val"])
    num_classes = len(train_dataset[0][1])
    print("Train_dataset length: ", len(train_dataset))
    print("Val_dataset length: ", len(val_dataset))
    print("Num_classes: ", num_classes)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    
    # 实例化模型
    model = SwinQ2L(num_classes=num_classes).to(device)

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

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = AsymmetricLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.2)
    print("Start training...")

    best_loss = torch.tensor(float('inf'))
    train_loss_list = []
    train_mAP_list = []
    loss_list = []
    mAP_list = []

    for epoch in range(args.epochs):
        # 进行训练和验证
        train_loss, train_mAP = train_one_epoch(model, optimizer, train_loader, device, epoch, criterion)
        val_loss, val_mAP = evaluate(model, val_loader, device, epoch, criterion)
        scheduler.step()

        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_mAP", train_mAP, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_mAP", val_mAP, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model": model.state_dict()}, "./weights/best.pth")
            print("Save best model at Epoch: {}".format(epoch))
        torch.save({"model": model.state_dict()}, "./weights/last.pth")
        loss_list.append(val_loss)
        mAP_list.append(val_mAP)
        train_loss_list.append(train_loss)
        train_mAP_list.append(train_mAP)

        # 保存loss和mAP
        with open("./weights/loss.txt", "w") as f:
            for loss in loss_list:
                f.write(str(loss) + "\n")
        with open("./weights/mAP.txt", "w") as f:
            for mAP in mAP_list:
                f.write(str(mAP) + "\n")
        with open("./weights/train_loss.txt", "w") as f:
            for loss in train_loss_list:
                f.write(str(loss) + "\n")
        with open("./weights/train_mAP.txt", "w") as f:
            for mAP in train_mAP_list:
                f.write(str(mAP) + "\n")
    
    tb_writer.close()
    print("Training finished.")

if __name__ == "__main__":
    args = parser_options()
    main(args)
