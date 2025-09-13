import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset

class PatientECGDatasetLoader(Dataset):
    
    def __init__(self, baseDir='', patients=[], normalize=True, normMethod='unitrange', rhythmType='Rhythm', numECGstoFind=1, spectogramConverter=None):
        self.baseDir = baseDir
        self.rhythmType = rhythmType
        self.normalize = normalize
        self.normMethod = normMethod
        self.fileList = []
        self.patientLookup = []
        self.spectogramConverter = spectogramConverter

        if len(patients) == 0:
            self.patients = os.listdir(baseDir)
        else:
            self.patients = patients
        
        if type(self.patients[0]) is not str:
            self.patients = [str(pat) for pat in self.patients]
        
        if numECGstoFind == 'all':
            for pat in self.patients:
                self.findEcgs(pat, 'all')
        else:
            for pat in self.patients:
                self.findEcgs(pat, numECGstoFind)
    
    def findEcgs(self, patient, numberToFind=1):
        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        numberOfEcgs = patientInfo['numberOfECGs']

        if(numberToFind == 1) | (numberOfEcgs == 1):
            for i in range(2):
                ecgId = str(patientInfo["ecgFileIds"][0])
                zeros = 5 - len(ecgId)
                ecgId = "0"*zeros+ ecgId
                self.fileList.append(os.path.join(patient,
                                    f'ecg_0',
                                    f'{ecgId}_{self.rhythmType}.npy'))
                self.patientLookup.append(f"{patient}_{i}")
        else:
            for ecgIx in range(numberOfEcgs):
                for i in range(2):
                    self.fileList.append(os.path.join(patient,
                                                f'ecg_{ecgIx}',
                                                f'{patientInfo["ecgFields"][ecgIx]}_{self.rhythmType}.npy'))
                    self.patientLookup.append(f"{patient}_{i}")
        
    
    def __getitem__(self, item):
        patient = self.patientLookup[item][:-2]
        segment = self.patientLookup[item][-1]

        patientInfoPath = os.path.join(self.baseDir, patient, 'patientData.json')
        patientInfo = json.load(open(patientInfoPath))
        
        ecgPath = os.path.join(self.baseDir,
                               self.fileList[item])
        
        ecgData = np.load(ecgPath)
        if(segment == '0'):
            ecgData = ecgData[:, 0:2500]
        else:
            ecgData = ecgData[:, 2500:]

        ejectionFraction = torch.tensor(patientInfo['ejectionFraction'])
        ecgs = torch.tensor(ecgData).float()

        if self.normalize:
            if self.normMethod == '0to1':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    ecgs = ecgs - torch.min(ecgs)
                    ecgs = ecgs / torch.max(ecgs)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
            elif self.normMethod == 'unitrange':
                if not torch.allclose(ecgs, torch.zeros_like(ecgs)):
                    for lead in range(ecgs.shape[0]):
                        frame = ecgs[lead]
                        frame = (frame - torch.min(frame)) / (torch.max(frame) - torch.min(frame) + 1e-8)
                        frame = frame - 0.5
                        ecgs[lead,:] = frame.unsqueeze(0)
                else:
                    print(f'All zero data for item {item}, {ecgPath}')
        
        if torch.any(torch.isnan(ecgs)):
            print(f"NANs in the data for item {item}, {ecgPath}")
        
        if self.spectogramConverter:
            ecgs = self.spectogramConverter(ecgs)
        
        return ecgs, ejectionFraction
    
    def __len__(self):
        return len(self.fileList)
