import click
import pandas as pd
import torch
import torch.optim as optim
from os import listdir
from math import log10
from tqdm import tqdm
from torch.utils.data import DataLoader
from util import TrainDatasetFromFolder, ValDatasetFromFolder, ssim
from module import Generator, Discriminator
from loss import GeneratorLoss
from transformers import ViTFeatureExtractor


@click.command()
@click.option('--epochs', default=100, help='Epochs')
@click.option('--crop_size', default=16, help='Crop size')
@click.option('--upscale_factor', default=2, help='Upscale factor')
@click.option('--train_dataset_dir', default='data/VOCdevkit/VOC2012/JPEGImages', help='Train directory')
@click.option('--val_dataset_dir', default='data/VOCdevkit/VOC2012/Val_Images', help='Validation directory')
@click.option('--save_dir', default='model/', help='Save directory')
def run(epochs, crop_size, upscale_factor, train_dataset_dir, val_dataset_dir, save_dir):
    # 1. Setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netG = Generator(upscale_factor)
    netD = Discriminator()

    generator_criterion = GeneratorLoss()

    netG.to(device)
    netD.to(device)
    generator_criterion.to(device)

    lr = 0.00008
    # adam: decay of first order momentum of gradient
    b1 = 0.5
    # adam: decay of second order momentum of gradient
    b2 = 0.999

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(b1, b2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(b1, b2))

    # 2. Generate DataLoader
    print(f'Train size: {len(listdir(train_dataset_dir))}, Validation size: {len(listdir(val_dataset_dir))}')

    train_set = TrainDatasetFromFolder(train_dataset_dir, crop_size=crop_size, upscale_factor=upscale_factor)
    val_set = ValDatasetFromFolder(val_dataset_dir, upscale_factor=upscale_factor)
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

    # 3. Train
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, epochs + 1):
        print()
        print(f'Epoch {epoch}/{epochs}')
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()

        # Train
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_img = target.to(device)
            z = data.to(device)

            # z = feature_extractor(images=data, return_tensors='pt')['pixel_values']
            # z = z.to(device)

            fake_img = netG(z)

            # Train Discriminator: Maximize D(x) -> 1 - D(G(z))
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            # Train Generator: Minimize 1 -> D(G(z)) + Content Loss + TV Loss
            netG.zero_grad()
            # PyTorch Inplace Operation Error
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='Loss_D: %.4f - Loss_G: %.4f - D(x): %.4f - D(G(z)): %.4f' % (
                running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        # Validate
        netG.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size

                lr = val_lr.to(device)
                hr = val_hr.to(device)

                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += (batch_mse * batch_size)  # 누적 mse

                batch_ssim = ssim(sr, hr).item()

                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(
                    (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='PSNR: %.4fdB - SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

        # Save model parameters
        torch.save(netG.state_dict(), f'{save_dir}my_model_epoch_{epoch}.pth')

        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'D(x)': results['d_score'],
                  'D(G(z))': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(1, epoch + 1))
        data_frame.to_csv(save_dir + 'train_results.csv', index_label='Epoch')


if __name__ == '__main__':
    run()
