import os

from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

SAMPLING_RATE = 500  
N_FFT = 128
HOP_LENGTH = 20
WIN_LENGTH = N_FFT 

def normalize(ecg):
    return (ecg - ecg.mean()) / ecg.std()


def convert_to_spectrogram(ecg):
    data = torch.stft(ecg, n_fft=N_FFT, hop_length=HOP_LENGTH,
                          win_length=WIN_LENGTH, window=torch.hann_window(WIN_LENGTH).to(ecg.device),
                          return_complex=True) # Calculate STFT
    data = torch.sqrt(data.real.pow(2) + data.imag.pow(2) + (1e-9)) # Calculate magntiude of the spectrogam
    data = torch.log(torch.clamp(data, min=1e-5) * 5) # Convert to decibel scale
    
    return data
    


class ECGToSpectrogramDataset(Dataset):
    def __init__(self, baseDir, ecgs, patientIds):
        self.baseDir = baseDir
        self.ecgs = ecgs
        self.patientIds = patientIds
    
    def __getitem__(self, item):
        ecg_idx = item // 8
        ecg_lead = item % 8

        ecgName = self.ecgs[ecg_idx].replace('.xml', '_Rhythm.npy')
        ecgPath = os.path.join(self.baseDir, ecgName)
        ecgData = np.load(ecgPath)

        ecg = torch.tensor(ecgData).float()
        ecg = ecg[ecg_lead] # Picking a lead

        # Cropping randomly to a fixed size of 2500 samples
        startIx = 0
        if ecg.shape[-1] - 2500 > 0:
            startIx = torch.randint(ecg.shape[-1]-2500, (1,))
        ecg = ecg[startIx:startIx+2500]

        # Normalize
        # ecg = normalize(ecg)
        
        # Convert To Spectrogram
        data = convert_to_spectrogram(ecg) 

        return (data, ecg)
    
    def __len__(self):
        return len(self.ecgs)*8


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

    # Split patients 90/10
    split_idx = int(0.9 * len(unique_patients))
    train_patients = unique_patients[:split_idx]
    # If scale_training_size is less than 1.0, adjust the training set size
    if scale_training_size < 1.0:
        train_patients = train_patients[:int(
            len(train_patients) * scale_training_size)]
    val_patients = unique_patients[split_idx:]

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
    train_dataset = ECGToSpectrogramDataset(
        baseDir=dataDir + 'pythonData/',
        ecgs=train_df['ECGFile'].tolist(),
        patientIds=train_df['PatId'].tolist(),
    )

    val_dataset = ECGToSpectrogramDataset(
        baseDir=dataDir + 'pythonData/',
        ecgs=val_df['ECGFile'].tolist(),
        patientIds=val_df['PatId'].tolist(),
    )

    return train_dataset, val_dataset        
        
         
        