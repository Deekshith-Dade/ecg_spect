import argparse
import random

import torch
import numpy as np
from sklearn import metrics

from data_utils import dataprepLVEF
from ecg_transforms import ECGToSpectogram
from model import ECGClassifier

from tqdm import tqdm
import logging


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Classification LVEF spectograms") 
parser.add_argument('--batch_size', default=1024, type=int, metavar='N', help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, metavar='N')
parser.add_argument('--epochs', default=100, type=int, help="Number of epochs to train")
parser.add_argument('--lr', default=1e-4, type=float, help='Learning Rate')


bce_loss = torch.nn.BCEWithLogitsLoss()

def normalize(train_loader):
    num_channels, height, width = train_loader.dataset[0][0].shape

    sum_val = torch.zeros(num_channels, dtype=torch.float64)
    sum_sq_val = torch.zeros(num_channels, dtype=torch.float64)
    
    # (num_samples * height * width)
    num_elements_per_channel = len(train_loader.dataset) * height * width

    for spectrograms, _ in tqdm(train_loader, desc="Calculating Mean and Std"):
        sum_val += torch.sum(spectrograms.to(torch.float64), dim=(0, 2, 3))
        sum_sq_val += torch.sum(spectrograms.to(torch.float64)**2, dim=(0, 2, 3))

    mean_per_channel = sum_val / num_elements_per_channel
    std_per_channel = torch.sqrt(
    (sum_sq_val / num_elements_per_channel) - (mean_per_channel**2)
    )

    print(f"Mean per channel: {mean_per_channel}")
    print(f"Std per channel: {std_per_channel}")
    
    logging.info(f"Mean {mean_per_channel}, Std: {std_per_channel}")

    return mean_per_channel.to(torch.float32), std_per_channel.to(torch.float32)


def evaluate(model, dataloader, lossFun, mean, std):
    model.eval()
    with torch.no_grad():
        running_loss = 0.
        allParams = torch.empty(0).to(device)
        allPredictions = torch.empty(0).to(device)
        
        for ecg, clinicalParam in dataloader:
            ecg = ecg.to(device)
            ecg = (ecg - mean) / std
            clinicalParam = clinicalParam.to(device).unsqueeze(1)
            predictedVal = model(ecg)
            lossVal = lossFun(predictedVal, clinicalParam)
            running_loss += lossVal.item()
            allParams = torch.cat((allParams, clinicalParam.squeeze()))
            allPredictions = torch.cat((allPredictions, predictedVal.squeeze()))
        
        running_loss = running_loss / len(dataloader)
    return running_loss, allParams, allPredictions
            


def main():
    args = parser.parse_args()
    epochs = args.epochs
    
    
    logging.basicConfig(
        filename='app.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    SAMPLING_RATE = 500  
    N_FFT = 128
    HOP_LENGTH = 20
    WIN_LENGTH = N_FFT 

    FILTER_ORDER = 4
    CUTOFF_FREQ = 40.0 
    
    spectogramConverter = ECGToSpectogram(SAMPLING_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH, apply_filter=False, output_size=256)
    
    train_dataloader, val_dataloader = dataprepLVEF(args, spectogramConverter)

    mean, std = normalize(train_dataloader)
    mean = mean.view(-1, 1, 1).to(device)
    std = std.view(-1, 1, 1).to(device)
    
    model = ECGClassifier(in_channels=3, num_classes=1)
    model.to(device)
    
    def loss_fn(predictedLogits, clinicalParam):
        clinicalParam = (clinicalParam < 40.0).float()
        return bce_loss(predictedLogits, clinicalParam)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    
    
    best_auc_test = 0.5
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        logging.info(f"Epoch {epoch+1} of {epochs}")
        model.train()
        count = 0 
        running_loss = 0
        for ecg, clinicalParam in train_dataloader:
            print(f'Running through training batches {count+1} of {len(train_dataloader)}', end='\r')
            
            count+=1
            optimizer.zero_grad()
            
            ecg = ecg.to(device)
            ecg = (ecg - mean) / std
            clinicalParam = clinicalParam.to(device).unsqueeze(1)
            predicted = model(ecg)
            loss = loss_fn(predicted, clinicalParam)
            
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        batch_total_loss = running_loss / len(train_dataloader)
        print()
        print(f"Batch Loss: {batch_total_loss}")
        logging.info(f"Batch Loss: {batch_total_loss}")
        
        print('Evalving Test')
        currTestLoss, allParams_test, allPredictions_test = evaluate(model, val_dataloader, loss_fn, mean, std)
        print('Evalving Train')
        currTrainLoss, allParams_train, allPredictions_train = evaluate(model, train_dataloader, loss_fn, mean, std)
        
        print(f"Train Loss: {currTrainLoss} \n Test Loss: {currTestLoss}")
        logging.info(f"Train Loss: {currTrainLoss} \n Test Loss: {currTestLoss}")
         
        
        
        allParams_train = (allParams_train.clone().detach().cpu() < 40.0).long().numpy()
        allPredictions_train = allPredictions_train.clone().detach().cpu().numpy()

        allParams_test = (allParams_test.clone().detach().cpu() < 40.0).long().numpy()
        allPredictions_test = allPredictions_test.clone().detach().cpu().numpy()

        # falsePos_train, truePos_train, _ = metrics.roc_curve(allParams_train, allPredictions_train)
        # falsePos_test, truePos_test, _ = metrics.roc_curve(allParams_test, allPredictions_test)
        auc_train = metrics.roc_auc_score(allParams_train, allPredictions_train)
        auc_test = metrics.roc_auc_score(allParams_test, allPredictions_test)
        
        if auc_test > best_auc_test:
            best_auc_test = auc_test
        
        print(f'Train AUC: {auc_train:0.6f} test AUC: {auc_test:0.6f}')
        print(f'best AUC Test: {best_auc_test:0.4f}')
        
        logging.info(f'Train AUC: {auc_train:0.6f} test AUC: {auc_test:0.6f}')
        logging.info(f'best AUC Test: {best_auc_test:0.4f}') 


if __name__ == "__main__":
    main()
