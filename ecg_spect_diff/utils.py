import math
import io

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

def plot_image_channels_grid(image_batch):
    image_batch = image_batch.cpu().permute(0, 2, 3, 1).numpy()
    
    num_images, height, width, num_channels = image_batch.shape
    
    fig, axes = plt.subplots(num_images, num_channels, figsize=(10, 3 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        for j in range(num_channels):
            ax = axes[i, j]
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
    
    return fig

def resize_and_stack_images(pil_images, size=256):
    transform_pipeline = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.ToTensor()
    ])

    tensor_batch = torch.stack([transform_pipeline(image) for image in pil_images])

    return tensor_batch