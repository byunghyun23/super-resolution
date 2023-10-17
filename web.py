import gradio as gr
import gc
import torch
from torchvision.transforms import ToTensor, ToPILImage, Resize
from module import Generator
from util import sliding_window, merge_patches_into_image


def upload_image(image):
    gc.collect()
    torch.cuda.empty_cache()

    # # SRGAN
    # image = ToTensor()(image).unsqueeze(0).to(device)
    # sr_image = my_model(image)
    # sr_image = ToPILImage()(sr_image[0].data.cpu())

    # SRGAN + Sliding Window
    patch_size = 16
    patch_size = patch_size // upscale_factor
    w, h = image.size

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

    patches = sliding_window(image, (patch_size, patch_size), stride)

    generated_patches = []
    for patch in patches:
        patch = ToTensor()(patch).unsqueeze(0).to(device)

        generated_patch = my_model(patch)
        generated_patch = ToPILImage()(generated_patch[0].data.cpu())

        generated_patches.append(generated_patch)

    patch_size *= upscale_factor
    stride = (stride[0] * upscale_factor, stride[1] * upscale_factor)
    sr_image = merge_patches_into_image(generated_patches, (w * upscale_factor, h * upscale_factor),
                                            (patch_size, patch_size), stride)

    return sr_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
upscale_factor = 2
model_name = 'model/my_model_epoch_100.pth'

my_model = Generator(upscale_factor).eval()
my_model.cuda()
my_model.load_state_dict(torch.load(model_name))

title = 'Super-Resolution'
description = 'Ref: https://github.com/byunghyun23/super-resolution'
image_input = gr.components.Image(label='Input image', type='pil')
output_image = gr.components.Image(label='Processed Image', type='pil')
custom_css = '#component-12 {display: none;} #component-1 {display: flex; justify-content: center; align-items: center;} img.svelte-ms5bsk {width: unset;}'

iface = gr.Interface(fn=upload_image, inputs=image_input, outputs=output_image,
                     title=title, description=description, css=custom_css)
iface.launch(server_port=8080)