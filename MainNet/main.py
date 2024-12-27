import os
import torch
import torch.nn as nn

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model import InvISPNet
from dataset import ShadowDataset
from train import train_model

import utils

def main():
    bs = 2 
    #im_size = 256
    color_pretrained_file = '../ColorTrans/checkpoints/checkpoint_499.pth'
    root_dir = '../../../datasets/ISTD_Dataset'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 900

    model = InvISPNet(color_pretrained_file)
    model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)

    criterion_l1 = nn.L1Loss().to(device)
    criterion_lab = utils.LabLoss(device).to(device)

    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
    optimizer = torch.optim.Adam(optim_params, lr=4e-4, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [300, 600], gamma=0.5)
    

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
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)
    dataloaders = {'train': train_loader, 'val': val_loader}

    save_dir = './checkpoints'

    train_model(model, dataloaders, criterion_l1, criterion_lab, optimizer, scheduler, num_epochs, device, save_dir)

if __name__ == '__main__':
    main()
