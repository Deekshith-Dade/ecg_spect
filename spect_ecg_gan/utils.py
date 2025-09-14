import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram(original, reproduced, prefix=""):
    fig, ax = plt.subplots(1, 2, figsize=(24, 6))
    im = ax[0].imshow(
        original,
        aspect='auto',
        origin='lower',
        cmap='viridis',
    )
    ax[0].set_title(f'{prefix} Original')
    im = ax[1].imshow(
        reproduced,
        aspect='auto',
        origin='lower',
        cmap='viridis',
    )
    ax[1].set_title(f'{prefix} Reproduced')

    fig.canvas.draw()
    plt.close()

    return fig

def plot_ecg_leads(original, reproduced, prefix=""):
    fig, axs = plt.subplots(figsize=(4*15, 4*2.5))
    axs.plot(list(range(reproduced.shape[-1])), original, linewidth=2, color='red')
    axs.plot(list(range(original.shape[-1])), reproduced, linewidth=1, color='blue', linestyle="--")
    axs.set_title(f'{prefix} leads')
    fig.canvas.draw()
    plt.close()
    return fig

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

