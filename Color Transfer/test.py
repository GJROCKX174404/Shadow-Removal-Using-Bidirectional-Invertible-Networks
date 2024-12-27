import os
import torch

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import utils
from dataset import ShadowDataset
from model import ConditionNet

from tqdm import tqdm

def test(model_path, results_dir, device):
    root_dir = '../../../datasets/ISTD_Dataset'
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = ShadowDataset(os.path.join(root_dir, 'test'), is_train=False, transform=val_transforms)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = ConditionNet()
    model = model.to(device)
    utils.load_model(model_path, model)
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Testing phase...')
        for images, masks, targets, img_names in pbar:
            images = images.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            outputs = model(images, masks)
            gen_imgs = utils.tensor2img(targets/(torch.mean(targets, 1).unsqueeze(1) + 1e-8))# gen_imgs will be 3 dimensional; h x w x 3 
            utils.save_img(gen_imgs, os.path.join(results_dir, '_1' +  img_names[0]))
            gen_imgs = utils.tensor2img(images/(torch.mean(images, 1).unsqueeze(1) + 1e-8))# gen_imgs will be 3 dimensional; h x w x 3 
            utils.save_img(gen_imgs, os.path.join(results_dir, '_2' + img_names[0]))
            assert False

if __name__ == '__main__':
    model_path = './checkpoints/checkpoint_999.pth'
    results_dir = './generated_images'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test(model_path, results_dir, device)
