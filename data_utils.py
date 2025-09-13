import random
import pickle
import DataTools
import torch
import time
import os
import numpy as np

def dataprepLVEF(args, spectogramConverter):
    dataDir = '/uufs/sci.utah.edu/projects/ClinicalECGs/LVEFCohort/pythonData/'
    normEcgs = False

    print("Preparing Data For Finetuning")
    with open('lvef_patients/validation_patients.pkl', 'rb') as file:
        validation_patients = pickle.load(file)
    
    with open('lvef_patients/training_patients.pkl', 'rb') as file:
        pre_train_patients = pickle.load(file)
    print(len(validation_patients))
    print(len(pre_train_patients))
    
    
    finetuning_patients = pre_train_patients 

    dataset = DataTools.PatientECGDatasetLoader(baseDir=dataDir, patients=finetuning_patients.tolist(), normalize=normEcgs, spectogramConverter=spectogramConverter)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    dataset_length = len(dataset)
    
    validation_dataset = DataTools.PatientECGDatasetLoader(baseDir=dataDir, patients=validation_patients.tolist(), normalize=normEcgs, spectogramConverter=spectogramConverter)
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print(f"Preparing Finetuning with {dataset_length} number of ECGs and validation with {len(validation_dataset)} number of ECGs")

    return train_loader, val_loader


def splitPatientsLVEF(seed):
    start = time.time()
    dataDir = '/uufs/sci.utah.edu/projects/ClinicalECGs/LVEFCohort/pythonData/'
    baseDir = '/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/lvef_patients'

    # Loading Data
    print('Finding Patients')
    patientIds = np.array(os.listdir(dataDir))
    numPatients = patientIds.shape[0]

    # Data
    pre_train_split_ratio = 0.9
    num_pre_train = int(pre_train_split_ratio * numPatients)
    num_validation = numPatients - num_pre_train

    patientInds = list(range(numPatients))
    random.Random(seed).shuffle(patientInds)

    pre_train_patient_indices = patientInds[:num_pre_train]
    validation_patient_indices = patientInds[num_pre_train:num_pre_train + num_validation]

    pre_train_patients = patientIds[pre_train_patient_indices].squeeze()
    validation_patients = patientIds[validation_patient_indices].squeeze()

    with open(f'{baseDir}/training_patients.pkl', 'wb') as file:
        pickle.dump(pre_train_patients, file)
    with open(f'{baseDir}/validation_patients.pkl', 'wb') as file:
        pickle.dump(validation_patients, file)
    print(f"Out of Total {numPatients} Splitting {len(pre_train_patients)} for pre-train and finetuning, {len(validation_patients)} for validation")
    print(f'The process took {time.time()-start} seconds')