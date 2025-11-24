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

class ECG_KCL_Datasetloader(Dataset):
    def __init__(self, baseDir='', ecgs=[], kclVals=[], low_threshold=4.0, high_threshold=5.0, rhythmType='Rhythm',
                 randomCrop=False, augmentation=False,
                 cropSize=2500, expectedTime=5000):
        self.baseDir = baseDir
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.rhythmType = rhythmType
        self.ecgs = ecgs
        self.kclVals = kclVals
        self.expectedTime = expectedTime
        self.cropSize = cropSize
        self.randomCrop = randomCrop
        self.use_latents = False
        if self.randomCrop:
            self.expectedTime = self.cropSize
        self.augs = Loader.TwoCropsTransform(transforms.Compose([
            Loader.SpatialTransform(),
        ]))
        self.resize = transforms.Resize((256, 256))

    def __getitem__(self, item):
        ecgName = self.ecgs[item].replace('.xml', f'_{self.rhythmType}.npy')
        ecgPath = os.path.join(self.baseDir, ecgName)
        ecgData = torch.tensor(np.load(ecgPath)).float()
        
        ecgs = self.augs(ecgData)[0]

        kclVal = torch.tensor(self.kclVals[item])
        

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
        item['y'] = 1 if kclVal <= self.high_threshold and kclVal >= self.low_threshold else 0
        item['key'] = 'kclVal'
        item['val'] = kclVal
        item['ecgPath'] = ecgPath
        cond_inputs = {}
        cond_inputs['class'] = item['y']
        if self.low_threshold <= kclVal <= self.high_threshold:
            condition_status = "Normal Signs of Hyperkalemia"
        elif self.high_threshold < kclVal <= self.high_threshold + 1:
            condition_status = "Early Stage Hyperkalemia"
        elif self.high_threshold + 1 < kclVal <= self.high_threshold + 2:
            condition_status = "Moderate Stage Hyperkalemia"
        else:
            condition_status = "Severe Stage Hyperkalemia"

        cond_inputs['text'] = f"ECG that shows {condition_status} with KCL value {kclVal:0.2f}"
        # cond_inputs['text'] = "an ECG spectrogram"
        item['cond_inputs'] = cond_inputs
        return item

    def __len__(self):
        return len(self.ecgs)

def getKCLTrainTestDataset(dataset_config):
    randSeed = dataset_config['randSeed']
    timeCutoff = dataset_config['timeCutOff']
    lowerCutoff = dataset_config['lowerCutOff']
    data_path = dataset_config['data_path']
    dataDir = dataset_config['dataDir']
    scale_training_size = dataset_config['scale_training_size']
    kclTaskParams = dataset_config['kcl_params']

    assert scale_training_size <= 1.0

    np.random.seed(randSeed)

    # kclCohort = np.load(dataDir+'kclCohort_v1.npy',allow_pickle=True)
    # data_types = {
    #     'DeltaTime': float,
    #     'KCLVal': float,
    #     'ECGFile': str,
    #     'PatId': int,
    #     'KCLTest': str
    # }
    kclCohort = pd.read_parquet(data_path)

    kclCohort = kclCohort[kclCohort['DeltaTime'] <= timeCutoff]
    kclCohort = kclCohort[kclCohort['DeltaTime'] > lowerCutoff]

    kclCohort = kclCohort.dropna(subset=['DeltaTime'])  # type: ignore
    kclCohort = kclCohort.dropna(subset=['KCLVal'])

    ix = kclCohort.groupby('ECGFile')['DeltaTime'].idxmin()
    kclCohort = kclCohort.loc[ix]

    numECGs = len(kclCohort)
    numPatients = len(np.unique(kclCohort['PatId']))

    print('setting up train/val split')
    numTest = int(0.1 * numPatients)
    numTrain = numPatients - numTest
    assert (numPatients == numTrain + numTest), "Train/Test spilt incorrectly"
    patientIds = list(np.unique(kclCohort['PatId']))
    random.Random(randSeed).shuffle(patientIds)

    trainPatientInds = patientIds[:numTrain]
    testPatientInds = patientIds[numTrain:numTest + numTrain]
    trainECGs = kclCohort[kclCohort['PatId'].isin(trainPatientInds)]
    testECGs = kclCohort[kclCohort['PatId'].isin(testPatientInds)]

    trainECGs = trainECGs[(trainECGs['KCLVal'] >= kclTaskParams['lowThresh']) & (
        trainECGs['KCLVal'] <= kclTaskParams['highThreshRestrict'])]
    testECGs = testECGs[(testECGs['KCLVal'] >= kclTaskParams['lowThresh']) & (
        testECGs['KCLVal'] <= kclTaskParams['highThreshRestrict'])]

    desiredTrainingAmount = int(len(trainECGs) * scale_training_size)
    print(f"{desiredTrainingAmount}: {len(trainECGs)}: {scale_training_size}")
    if desiredTrainingAmount != 'all':
        if len(trainECGs) > desiredTrainingAmount:
            trainECGs = trainECGs.sample(n=desiredTrainingAmount)

    dataset_regular = ECG_KCL_Datasetloader
    trainDataset = dataset_regular(
        baseDir=dataDir + 'pythonData/',
        ecgs=trainECGs['ECGFile'].tolist(),
        low_threshold=kclTaskParams['lowThresh'],
        high_threshold=kclTaskParams['highThresh'],
        kclVals=trainECGs['KCLVal'].tolist(),
        randomCrop=True
    )
    print(f'Number of Training Examples: {len(trainDataset)}')

    testDataset = dataset_regular(
        baseDir=dataDir + 'pythonData/',
        ecgs=testECGs['ECGFile'].tolist(),
        low_threshold=kclTaskParams['lowThresh'],
        high_threshold=kclTaskParams['highThresh'],
        kclVals=testECGs['KCLVal'].tolist(),
        randomCrop=True
    )

    return trainDataset, testDataset

class SpectrogramExtractor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.leads = 8
        self.orig_height = 64
        self.orig_width = 126
        
        self.grid_rows = 4
        self.grid_cols = 2
        
        self.pad_left = 2
        self.pad_right = 2
        self.resize = transforms.Resize((256, 256))
    
    def forward(self, x):
        
        x = self.resize(x)
        unpadded_width = x.shape[-1] - self.pad_left - self.pad_right
        x = x[:, :, self.pad_left : self.pad_left + unpadded_width] # 8 x 256 x 252
        
        B = x.shape[0]
        grid = x.view(B, self.grid_rows, self.orig_height, self.grid_cols, self.orig_width) 
        # 8 x 4 x 64 x 2 x 126
        
        spectrograms = grid.permute(0, 1, 3, 2, 4)
        # 8 x 4 x 2 x 64 x 126
        
        final_shape = (B, self.leads, self.orig_height, self.orig_width) 
        spectrograms = spectrograms.reshape(final_shape)
        spectrograms = F.pad(spectrograms, (0, 0, 0, 1), mode="constant", value=0)
        return spectrograms
            