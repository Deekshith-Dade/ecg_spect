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
    n_ecgs, leads, ti = ecgs.shape
    fig, axs = plt.subplots(8, figsize=(4*15, 4*leads*2.5))
    fig.suptitle(f'{label}', fontsize=50, y=0.92)

    colors = ['red', 'green', 'blue']

    for i in range(n_ecgs):
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

def plot_image_channels_grid(image_batch):
    image_batch = image_batch.cpu().permute(0, 2, 3, 1).numpy()
    
    num_images, height, width, num_channels = image_batch.shape
    
    full_figs = []
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        fig, axes = plt.subplots(1, num_channels, figsize=(3 * 8 + 6, 8))
        for j in range(num_channels):
            ax = axes[j]
            ax.imshow(image_batch[i, :, :, j],
                      aspect='auto',
                    origin='lower',
                    cmap='viridis'
                        )
            ax.set_title(f'Image {i+1}, Chanenel {j+1}')
            ax.axis('off')
        plt.tight_layout(pad=0.5)
        fig.canvas.draw()
        plt.close()
        full_figs.append(fig)
    
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