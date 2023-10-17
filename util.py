import torch
import torch.nn.functional as F
from os.path import join
from os import listdir
from PIL import Image
from math import exp
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        try:
            hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
            lr_image = self.lr_transform(hr_image)
        except Exception as e:
            print(self.image_filenames[index])
        # print(ToPILImage()(lr_image.data.cpu()).size)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.Resampling.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.Resampling.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([RandomCrop(crop_size), ToTensor()])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([ToPILImage(), Resize(crop_size // upscale_factor, interpolation=Image.Resampling.BICUBIC), ToTensor()])


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def sliding_window(image, patch_size, stride):
    patches = []
    height, width = image.size[1], image.size[0]

    for y in range(0, height - patch_size[1] + 1, stride[1]):
        for x in range(0, width - patch_size[0] + 1, stride[0]):
            patch = image.crop((x, y, x + patch_size[0], y + patch_size[1]))
            patches.append(patch)

    return patches


def merge_patches_into_image(patches, image_size, patch_size, stride):
    num_rows = (image_size[1] - patch_size[1]) // stride[1] + 1
    num_cols = (image_size[0] - patch_size[0]) // stride[0] + 1

    output_image = Image.new('RGB', image_size)
    count_map = Image.new('L', image_size)
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            x = j * stride[0]
            y = i * stride[1]

            # Merge
            output_image.paste(patches[idx], (x, y))

            # Accumulation
            count_map.paste(Image.new('L', patch_size, 255), (x, y))

            idx += 1

    # Average image generation
    output_image = Image.composite(output_image, Image.new('RGB', image_size, (0, 0, 0)), count_map)

    return output_image