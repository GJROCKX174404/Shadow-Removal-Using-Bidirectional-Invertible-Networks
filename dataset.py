import os
import random
import torchvision.transforms.functional as TF

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class ShadowDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        train_or_test_folder = root_dir.split('/')[-1]
        if is_train:
            assert train_or_test_folder=='train' 
        else:
            assert train_or_test_folder=='test'
        self.image_folder = os.path.join(root_dir, train_or_test_folder + '_A')
        self.mask_folder = os.path.join(root_dir, train_or_test_folder + '_B')
        self.gt_folder = os.path.join(root_dir, train_or_test_folder + '_C')

        # Make sure the folders exist
        assert os.path.exists(self.image_folder), f"Images folder '{self.image_folder}' does not exist."
        assert os.path.exists(self.mask_folder), f"Mask folder '{self.mask_folder}' does not exist."
        assert os.path.exists(self.gt_folder), f"Groundtruth folder '{self.gt_folder}' does not exist."

        # List all files in the images folder
        self.image_files = os.listdir(self.image_folder)

    def transform(self, image, mask, target):
        """# Resize
        resize = transforms.Resize(size=(256, 256))
        image = resize(image)
        mask = resize(mask)
        target = resize(target)"""

        """# Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(512, 512))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        target = TF.crop(target, i, j, h, w)"""

        # Random horizontal flipping
        if self.is_train and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            target = TF.hflip(target)

        # Random vertical flipping
        if self.is_train and random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            target = TF.vflip(target)

        # Random rotate by 180 degrees
        if self.is_train and random.random() > 0.5:
            image = TF.rotate(image, 180, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, 180, interpolation=transforms.InterpolationMode.BILINEAR)
            target = TF.rotate(target, 180, interpolation=transforms.InterpolationMode.BILINEAR)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        target = TF.to_tensor(target)

        return image, mask, target

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        mask_name = os.path.join(self.mask_folder, self.image_files[idx])
        gt_name = os.path.join(self.gt_folder, self.image_files[idx])

        # Open images using PIL
        image = Image.open(img_name)
        mask = Image.open(mask_name)
        ground_truth = Image.open(gt_name)
        
        # Apply transformations
        image, mask, ground_truth = self.transform(image, mask, ground_truth)

        return image, mask, ground_truth, self.image_files[idx]

if __name__ == '__main__':
    # Define transformations
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])

    # Create an instance of the ShadowDataset
    root_dir = '/home/darshan/bala/shadow_removal/datasets/ISTD_Dataset/train'
    shadow_dataset = ShadowDataset(root_dir=root_dir, transform=data_transforms)

    # Accessing a sample from the dataset
    sample = shadow_dataset[0]
    print(f"{sample[0].shape}, {sample[1].shape}, {sample[2].shape}")
