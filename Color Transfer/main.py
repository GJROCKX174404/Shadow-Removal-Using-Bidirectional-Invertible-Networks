import os
import torch
import torch.nn as nn

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model import ConditionNet
from dataset import ShadowDataset
from train import train_model

def main():
    bs = 8
    im_size = 256
    root_dir = '../../../datasets/ISTD_Dataset'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 1200

    model = ConditionNet(channels=8)
    model = model.to(device)

    criterion = nn.L1Loss()
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [250, 500, 750, 1000], gamma=0.5)
    

    """train_transforms = transforms.Compose([
        #transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees = (180, 180), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    val_transforms = transforms.Compose([
        #transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])"""

    train_dataset = ShadowDataset(os.path.join(root_dir, 'train'), is_train=True)
    val_dataset = ShadowDataset(os.path.join(root_dir, 'test'), is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                           num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader}

    save_dir = './checkpoints'

    train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, save_dir)

if __name__ == '__main__':
    main()
