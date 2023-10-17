import click
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage, Resize
from module import Generator
from util import sliding_window, merge_patches_into_image


@click.command()
@click.option('--upscale_factor', default=2, help='Upscale factor')
@click.option('--patch_size', default=16, help='Patch size')
@click.option('--file_name', default='sample.jpg', help='Input file name')
@click.option('--model_name', default='model/my_model_epoch_100.pth', help='Model name')
def run(upscale_factor, patch_size, file_name, model_name):
    # 1. Setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Load model
    my_model = Generator(upscale_factor).eval()
    my_model.cuda()
    my_model.load_state_dict(torch.load(model_name))

    # 3. Load input image
    input_image = Image.open(file_name)

    # 4. SRGAN
    sr_image = input_image.copy()
    # sr_image = Resize(patch_size, interpolation=Image.Resampling.BICUBIC)(sr_image)
    sr_image = ToTensor()(sr_image).unsqueeze(0).to(device)
    sr_image = my_model(sr_image)
    sr_image = ToPILImage()(sr_image[0].data.cpu())

    # 5. SRGAN + Sliding Window
    patch_size = patch_size // upscale_factor
    output_image = input_image.copy()

    w, h = output_image.size

    x_stride = patch_size
    y_stride = patch_size

    target_height = h - patch_size
    for i in range(target_height, 0, -1):
        if target_height % i == 0 and i < patch_size // 2:
            x_stride = i
            break
    target_width = w - patch_size
    for i in range(target_width, 0, -1):
        if target_width % i == 0 and i < patch_size // 2:
            y_stride = i
            break

    stride = (x_stride, y_stride)

    patches = sliding_window(output_image, (patch_size, patch_size), stride)

    generated_patches = []
    for patch in patches:
        patch = ToTensor()(patch).unsqueeze(0).to(device)

        generated_patch = my_model(patch)
        generated_patch = ToPILImage()(generated_patch[0].data.cpu())

        generated_patches.append(generated_patch)

    patch_size *= upscale_factor
    stride = (stride[0] * upscale_factor, stride[1] * upscale_factor)
    output_image = merge_patches_into_image(generated_patches, (w * upscale_factor, h * upscale_factor),
                                            (patch_size, patch_size), stride)

    # 5. Visualize the input image and output image
    plt.subplot(1, 3, 1)
    plt.imshow(input_image)
    plt.title('INPUT')
    plt.axis('off')

    output_image.save('srgan.jpg', format='JPEG', quality=95)
    plt.subplot(1, 3, 2)
    plt.imshow(sr_image)
    plt.title('SRGAN')
    plt.axis('off')

    output_image.save('output.jpg', format='JPEG', quality=95)
    plt.subplot(1, 3, 3)
    plt.imshow(output_image)
    plt.title('SRGAN + Sliding Window')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()