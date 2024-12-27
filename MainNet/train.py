import os
import torch
import torch.nn as nn

from tqdm import tqdm  # Import tqdm for progress bars
#from piq import ssim
import utils

def train_model(model, dataloaders, criterion_l1, criterion_lab, optimizer, scheduler, num_epochs, device, save_dir):
    best_metric = float('-inf')  # Initialize the best metric to a very low value
    best_epoch = -1

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_metric = 0.0

            # Use tqdm to display progress bar
            data_loader = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Batch')

            # Iterate over data
            for images, masks, targets, img_names in data_loader:
                images = images.to(device)
                masks = masks.to(device)
                #targets = targets / (torch.mean(targets,1).unsqueeze(1)+1e-8) # this is the color map of ground-truth that will act as target
                targets = targets.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, mask_color = model(images, masks, targets.detach(), rev=False)
                    outputs_rev, _ = model(images, masks, targets.detach(), rev=True)
                    loss = criterion_l1(outputs, targets) + 0.4 * criterion_lab(outputs, targets.detach()) + 0.1 * criterion_l1(outputs_rev, images.detach())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                if '88-16.png' in img_names:
                    index = img_names.index('88-16.png')
                    gen_img = utils.tensor2img(outputs.detach()[index].unsqueeze(1))# gen_imgs will be 3 dimensional; h x w x 3 
                    utils.save_img(gen_img, os.path.join('./generated_images', img_names[index]))
                    gen_img = utils.tensor2img(targets.detach()[index].unsqueeze(1))# gen_imgs will be 3 dimensional; h x w x 3 
                    utils.save_img(gen_img, os.path.join('./generated_images', 'target'+img_names[index]))
                if '133-25.png' in img_names:
                    index = img_names.index('133-25.png')
                    gen_img = utils.tensor2img(outputs.detach()[index].unsqueeze(1))# gen_imgs will be 3 dimensional; h x w x 3 
                    utils.save_img(gen_img, os.path.join('./generated_images', img_names[index]))
                    gen_img = utils.tensor2img(targets.detach()[index].unsqueeze(1))# gen_imgs will be 3 dimensional; h x w x 3 
                    utils.save_img(gen_img, os.path.join('./generated_images', 'target'+img_names[index]))
                if '116-1.png' in img_names:
                    index = img_names.index('116-1.png')
                    gen_img = utils.tensor2img(outputs.detach()[index].unsqueeze(1))# gen_imgs will be 3 dimensional; h x w x 3 
                    utils.save_img(gen_img, os.path.join('./generated_images', img_names[index]))
                    gen_img = utils.tensor2img(targets.detach()[index].unsqueeze(1))# gen_imgs will be 3 dimensional; h x w x 3 
                    utils.save_img(gen_img, os.path.join('./generated_images', 'target'+img_names[index]))

                # Compute the metric (e.g., accuracy) for this batch
                #batch_metric = ssim((outputs - outputs.min()) / (outputs.max() - outputs.min()), (targets - targets.min()) / (targets.max() - targets.min()))
                batch_metric = utils.psnr_np(outputs.detach(), targets.detach())
                running_loss += loss.item() * images.size(0)
                running_metric += batch_metric.item() * images.size(0)

                # Update tqdm progress bar
                data_loader.set_postfix(loss=loss.item(), metric=batch_metric.item())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_metric = running_metric / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}, Metric: {epoch_metric:.4f}')

            # If this is the validation phase and the metric is better than the best so far, save the model
            if (phase == 'val' and epoch_metric > best_metric) or (phase == 'train' and epoch==300) or (phase=='train' and epoch==600) or (phase=='train' and epoch==899):
                if phase == 'val':
                    print("Best model found! Saving...")
                    best_metric = epoch_metric
                    best_epoch = epoch

                # Save model, optimizer state_dict & epoch number
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                save_str = '_best' if phase == 'val' else f'_{epoch}'
                save_checkpoint(checkpoint, save_dir, save_str)

        
        # Update the learning rate scheduler
        if scheduler:
            scheduler.step()
    print(f"Training complete. Best model found in epoch {best_epoch + 1}.")

def save_checkpoint(checkpoint, save_dir, save_str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'checkpoint' + save_str + '.pth')
    torch.save(checkpoint, save_path)

