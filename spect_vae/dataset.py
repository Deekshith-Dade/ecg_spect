import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

import ecg_spect_diff.dataset.aug_loader as Loader

class DataLoaderError(Exception):
    pass

SAMPLING_RATE = 500  
N_FFT = 128
HOP_LENGTH = 20
WIN_LENGTH = N_FFT 

def convert_to_spectrogram(ecg):
    data = torch.stft(ecg, n_fft=N_FFT, hop_length=HOP_LENGTH,
                          win_length=WIN_LENGTH, window=torch.hann_window(WIN_LENGTH).to(ecg.device),
                          return_complex=True) # Calculate STFT
    data = torch.sqrt(data.real.pow(2) + data.imag.pow(2) + (1e-9)) # Calculate magntiude of the spectrogam
    data = torch.log(torch.clamp(data, min=1e-5) * 5) # Convert to decibel scale
    return data

class ECG_Datasetloader(Dataset):
    def __init__(self, baseDir='', ecgs=[], rhythmType='Rhythm',
                 randomCrop=True, augmentation=False,
                 cropSize=2500, expectedTime=5000):
        self.baseDir = baseDir
        self.rhythmType = rhythmType
        self.ecgs = ecgs
        # self.kclVals = kclVals
        self.expectedTime = expectedTime
        self.cropSize = cropSize
        self.randomCrop = randomCrop
        self.use_latents = False
        if self.randomCrop:
            self.expectedTime = self.cropSize
        # self.augs = Loader.TwoCropsTransform(transforms.Compose([
        #     Loader.SpatialTransform(),
        # ]))
        self.resize = transforms.Resize((256, 256))

    def __getitem__(self, item):
        ecgName = self.ecgs[item].replace('.xml', f'_{self.rhythmType}.npy')
        ecgPath = os.path.join(self.baseDir, ecgName)
        ecgs = torch.tensor(np.load(ecgPath)).float()
        
        
        # ecgs = self.augs(ecgData)[0]

        if self.randomCrop:
            startIx = 0
            if ecgs.shape[-1]-self.cropSize > 0:
                startIx = torch.randint(ecgs.shape[-1] - self.cropSize, (1,))
            ecgs = ecgs[..., startIx:startIx+self.cropSize]

        if torch.any(torch.isnan(ecgs)):
            print(f'Nans in the data for item {item}, {ecgPath}')
            raise DataLoaderError('Nans in data')

            
        # Convert ecg to spectrogram and make it an image
        spect = convert_to_spectrogram(ecgs)    # 8x65x126
        spect = spect[:, :64, :]                # 8x64x126
        grid = spect.view(4, 2, 64, 126)
        grid = grid.permute(0, 2, 1, 3)         # 4 x 64 x 2 x 126
        img = grid.reshape(256, 252)
        img = F.pad(img, (2, 2), "constant", 0) # 256 x 256
        img = img.expand(3, -1, -1)             # 3 x 256 x 256
        img = self.resize(img)
        
        item = {}
        item['image'] = img
        item['ecgs'] = ecgs
        
        return item

    def __len__(self):
        return len(self.ecgs)

def get_datasets(scale_training_size=1.0):

    dataDir = '/uufs/sci.utah.edu/projects/ClinicalECGs/AllClinicalECGs/'
    print('finding patients')
    df = pd.read_csv(
        '/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/spect_ecg_gan/csv/ecgs_patients_mod.csv')
    # df.to_csv('ecg_files_df.csv', index=False)
    print(f"Number of ECGs: {len(df)}")

    # Get unique patient IDs
    unique_patients: np.ndarray = df['PatId'].unique()  # type: ignore
    np.random.shuffle(unique_patients)  # Shuffle for random split

    # Split patients 99/01
    split_idx = int(0.99 * len(unique_patients))
    train_patients = unique_patients[:split_idx]
    # If scale_training_size is less than 1.0, adjust the training set size
    if scale_training_size < 1.0:
        train_patients = train_patients[:int(
            len(train_patients) * scale_training_size)]
    val_patients = unique_patients[split_idx:]
    val_patients = val_patients[:int(len(val_patients)*0.5)]

    print(f"Total patients: {len(unique_patients)}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")

    # Split dataframe based on patient IDs
    train_df = df[df['PatId'].isin(
        train_patients.tolist())].reset_index(drop=True)
    val_df = df[df['PatId'].isin(val_patients.tolist())].reset_index(drop=True)

    print(f"Train ECGs: {len(train_df)}")
    print(f"Validation ECGs: {len(val_df)}")

    # Create datasets
    train_dataset = ECG_Datasetloader(
        baseDir=dataDir + 'pythonData/',
        ecgs=train_df['ECGFile'].tolist(),
    )

    val_dataset = ECG_Datasetloader(
        baseDir=dataDir + 'pythonData/',
        ecgs=val_df['ECGFile'].tolist(),
    )

    return train_dataset, val_dataset   