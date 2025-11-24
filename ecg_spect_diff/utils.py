import math
import io
import json

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

from spect_ecg_gan.models import Generator
from spect_ecg_gan.env import AttrDict

def load_generator_model(checkpoint_path):
    baseDir = "/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/"
    folder =  checkpoint_path.split("/")[2]
    config_file_path = f"{baseDir}/spect_ecg_gan/checkpoints/{folder}/config.json"
    with open(config_file_path) as f:
        data = f.read()
    
    json_config = json.loads(data)
    h = AttrDict(json_config)

    state_dict_g = torch.load(f"{baseDir}/{checkpoint_path}", map_location=torch.device("cpu"))
    generator = Generator(h)
    generator.load_state_dict(state_dict_g['generator'], strict=True)
    return generator

def plot_overlapping_ecgs(ecgs, label):
    # ecgs should be of shape (n, 8, 2500)
    ecgs = ecgs.detach().cpu()
    n_ecgs, leads, ti = ecgs.shape
    fig, axs = plt.subplots(8, figsize=(4*15, 4*leads*2.5))
    fig.suptitle(f'{label}', fontsize=50, y=0.92)

    colors = ['red', 'green', 'blue', 'yellow']
    if n_ecgs == 4:
        colors = ['black','red', 'green', 'blue']
    if n_ecgs == 5:
        colors = ['orange', 'black', 'red', 'green', 'blue']

    for i in range(n_ecgs):
        if i >= 3:
            pass
        color = colors[i%len(colors)]
        for lead in range(leads):
            y = list(ecgs[i, lead, :])
            axs[lead].plot(list(range(ti)), y, linewidth=2, color=color)
     
    for lead in range(leads):
        axs[lead].set_xlabel(f'Lead {lead+1}', fontsize=10)   
        axs[lead].xaxis.label.set_visible(True)
        axs[lead].tick_params(axis='x', labelsize=20)
        axs[lead].tick_params(axis='y', labelsize=20)
    
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    fig.canvas.draw()
    plt.close()
    return fig

def plot_image_channels_grid(image_batch, prefix=None, stack=False):
    image_batch = image_batch.cpu().permute(0, 2, 3, 1).numpy()
    
    num_images, height, width, num_channels = image_batch.shape

    if isinstance(prefix, str):
        prefix = [prefix] * num_images
    elif isinstance(prefix, list):
        assert len(prefix) == num_images
    
    full_figs = []
    if stack:
        fig, axes = plt.subplots(num_images, num_channels, figsize = (3 * 8 + 6, num_images * 8 + 6))
    for i in range(num_images):
        if not stack:
            fig, axes = plt.subplots(1, num_channels, figsize=(3 * 8 + 6, 8))
        for j in range(num_channels):
            if not stack:
                ax = axes[j]
            else:
                ax = axes[i, j]
            ax.imshow(image_batch[i, :, :, j],
                      aspect='auto',
                    origin='lower',
                    cmap='viridis'
                        )
            ax.set_title(f'Image {i+1}, Channel {j+1}')
            ax.axis('off')
        if not stack:
            plt.tight_layout(pad=0.5)
            # plt.title(prefix[i])
            fig.canvas.draw()
            plt.close()
            full_figs.append(fig)
    if stack:
        plt.tight_layout(pad=0.5)
        fig.canvas.draw()
        plt.close()
        full_figs = fig

    return full_figs

def resize_and_stack_images(pil_images, size=256):
    
    transform_pipeline = transforms.Compose([
        # transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC,
        #                   antialias=True),
        transforms.ToTensor()
    ])

    tensor_batch = torch.stack([transform_pipeline(image) for image in pil_images])
    # tensor_batch = tensor_batch[:,:,:size, :size]

    return tensor_batch

def plot_spectrograms_comparison(og_img, compare_images, og_prompt, compare_prompts):
    if len(og_img.shape) != len(compare_images.shape):
        og_img = og_img.unsqueeze(0)

    prompts = [og_prompt] + [compare_prompts] * compare_images.shape[0]
    
    fig = plot_image_channels_grid(torch.cat([og_img, compare_images]), prefix=prompts, stack=True)
    return fig