import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from sklearn import metrics
import logging

bce_loss = nn.BCELoss()

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available, skipping wandb logging")

def loss_bce_kcl(predictedVal,clinicalParam,lossParams):
	# clinicalParam = ((clinicalParam <= lossParams['highThresh']) * (clinicalParam >= lossParams['lowThresh'])).float()
	return bce_loss(predictedVal,clinicalParam)

def evaluate_balanced(network, dataLoaders, lossFun, lossParams, leads, device, 
                     ddp_enabled=False, lookForFig=False):
    network.eval()
    # plt.figure(2)
    # fig, ax1 = plt.subplots(8,2, figsize=(4*15, 4*8*2.5))
    pltCol = 0
					
    with torch.no_grad():
        running_loss = 0.
		
        allParams = torch.empty(0).to(device)
        allPredictions = torch.empty(0).to(device)
        for dataLoader in dataLoaders:
            for ecg, clinicalParam in dataLoader:
                ecg = ecg[:,leads,:].to(device)
                clinicalParam = clinicalParam.to(device).unsqueeze(1) 
                predictedVal = network(ecg)
                lossVal = lossFun(predictedVal,clinicalParam,lossParams)
                running_loss += lossVal.item()

                allParams = torch.cat((allParams, clinicalParam.squeeze()) )
                allPredictions = torch.cat((allPredictions, predictedVal.squeeze()))
                if lookForFig:
                    binaryParams = ((clinicalParam <= lossParams['highThresh']) * (clinicalParam >= lossParams['lowThresh'])).float()
                    agreement = torch.abs(binaryParams.squeeze()-predictedVal.squeeze())
                    ecgIx = torch.argmax(agreement)
                    disagree_kcl = clinicalParam[ecgIx,...]
                    disagree_pred = allPredictions[ecgIx,...]
                    for lead in range(8):
                        ax1[lead,pltCol].plot(ecg[ecgIx,lead,:].detach().clone().squeeze().cpu().numpy(),'k')
                    ecgIx = torch.argmin(agreement)
                    agree_kcl = clinicalParam[ecgIx,...]
                    agree_pred = allPredictions[ecgIx,...]
                    for lead in range(8):
                        ax1[lead,pltCol+1].plot(ecg[ecgIx,lead,:].detach().clone().squeeze().cpu().numpy(),'k')
                    lookForFig = False
                    ax1[0,pltCol].text(0,100, f'Disagree: {disagree_kcl.item()}, pred: {disagree_pred}.')
                    ax1[0,pltCol+1].text(0, 100,f'Agree: {agree_kcl.item()}, pred: {agree_pred}.')
                    fig.suptitle(f'Disagree: {disagree_kcl.item()}, pred: {disagree_pred}.\nAgree: {agree_kcl.item()}, pred: {agree_pred}.', fontsize=50, y=0.95)
                    print(f'Disagree: {disagree_kcl.item()}, pred: {disagree_pred}.\nAgree: {agree_kcl.item()}, pred: {agree_pred}.')		
		
        # Gather results from all processes if DDP is enabled
        if ddp_enabled and dist.is_initialized():
            # Get world size
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            # Gather sizes from all processes first
            local_size = torch.tensor(allParams.shape[0], dtype=torch.long, device=device)
            sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(sizes, local_size)
            sizes = [s.item() for s in sizes]
            max_size = max(sizes) if sizes else allParams.shape[0]
            
            # Pad tensors to max_size for all_gather (all_gather requires same size tensors)
            if allParams.shape[0] < max_size:
                padding_size = max_size - allParams.shape[0]
                # Use a sentinel value or zeros for padding (zeros should be fine since we'll slice)
                allParams_padded = torch.cat([allParams, torch.zeros(padding_size, device=device, dtype=allParams.dtype)])
                allPredictions_padded = torch.cat([allPredictions, torch.zeros(padding_size, device=device, dtype=allPredictions.dtype)])
            else:
                allParams_padded = allParams
                allPredictions_padded = allPredictions
            
            # Gather from all processes
            gathered_params_list = [torch.zeros_like(allParams_padded) for _ in range(world_size)]
            gathered_predictions_list = [torch.zeros_like(allPredictions_padded) for _ in range(world_size)]
            dist.all_gather(gathered_params_list, allParams_padded)
            dist.all_gather(gathered_predictions_list, allPredictions_padded)
            
            # Concatenate and remove padding
            allParams_list = []
            allPredictions_list = []
            for i in range(world_size):
                actual_size = sizes[i]
                allParams_list.append(gathered_params_list[i][:actual_size])
                allPredictions_list.append(gathered_predictions_list[i][:actual_size])
            
            allParams = torch.cat(allParams_list)
            allPredictions = torch.cat(allPredictions_list)
            
            # Gather and average loss across processes
            loss_tensor = torch.tensor(running_loss, dtype=torch.float32, device=device)
            gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_losses, loss_tensor)
            # Average the loss (each process computed loss on its subset)
            running_loss = sum([l.item() for l in gathered_losses]) / world_size
        else:
            # Non-DDP: average loss across data loaders
            running_loss = running_loss / len(dataLoaders) if len(dataLoaders) > 0 else running_loss
        
        # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3)
        plot_margin = 0.25

        # x0, x1, y0, y1 = plt.axis()
        # plt.axis((x0 - plot_margin,
        #         x1 + plot_margin,
        #         y0 - plot_margin,
        #         y1 + plot_margin))

        return running_loss, allParams, allPredictions, None


def trainNetwork_balancedClassification(network, trainDataLoader_normals, trainDataLoader_abnormals, testDataLoader, numEpoch, optimizer, lossFun, lossParams,leads,modelSaveDir, modelName, logger=None, logToWandB=False, device=None, ddp_enabled=False, is_main_process=True, model_for_saving=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Set default device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use model_for_saving if provided (for DDP), otherwise use network directly
    if model_for_saving is None:
        model_for_saving = network
    
    if is_main_process:
        logger.info(f"Beginning Training for Network {network.__class__.__name__}")
        if ddp_enabled:
            logger.info(f"DDP enabled with {dist.get_world_size()} processes")
    prevTrainingLoss = 0.0
    exhausted = 0
    bestEvalMetric_test = 0.5
    best_acc = 0.5
    maxBatches = max(len(trainDataLoader_normals), len(trainDataLoader_abnormals))

    for ep in range(numEpoch):
        if is_main_process:
            logger.info(f"Epoch {ep+1} of {numEpoch}")
        
        # Set epoch for distributed samplers
        if ddp_enabled and dist.is_initialized():
            if hasattr(trainDataLoader_normals, 'sampler') and hasattr(trainDataLoader_normals.sampler, 'set_epoch'):
                trainDataLoader_normals.sampler.set_epoch(ep)
            if hasattr(trainDataLoader_abnormals, 'sampler') and hasattr(trainDataLoader_abnormals.sampler, 'set_epoch'):
                trainDataLoader_abnormals.sampler.set_epoch(ep)
            if hasattr(testDataLoader, 'sampler') and hasattr(testDataLoader.sampler, 'set_epoch'):
                testDataLoader.sampler.set_epoch(ep)
        
        running_loss = 0.0
        network.train()

        iter_normal = iter(trainDataLoader_normals)
        iter_abnormal = iter(trainDataLoader_abnormals)

        for batchIx in range(maxBatches):
            try:
                ecg_normal, clinicalParam_normal = next(iter_normal)
            except:
                iter_normal = iter(trainDataLoader_normals)
                ecg_normal, clinicalParam_normal = next(iter_normal)
            
            try:
                ecg_abnormal, clinicalParam_abnormal = next(iter_abnormal)
            except:
                iter_abnormal = iter(trainDataLoader_abnormals)
                ecg_abnormal, clinicalParam_abnormal = next(iter_abnormal)
            
            numNormal = ecg_normal.shape[0]
            numAbnormal = ecg_abnormal.shape[0]

            ecg = torch.empty((numNormal+numAbnormal, *list(ecg_normal.shape[1:])))
            clincalParam = torch.empty((numNormal+numAbnormal, *list(clinicalParam_normal.shape[1:])))

            shuffleIxs = torch.randperm(numNormal+numAbnormal)
            normIxs = shuffleIxs[:numNormal]
            abnormIxs = shuffleIxs[numNormal:]

            ecg[normIxs,...] = ecg_normal
            clincalParam[normIxs,...] = clinicalParam_normal
            ecg[abnormIxs,...] = ecg_abnormal
            clincalParam[abnormIxs,...] = clinicalParam_abnormal

            if batchIx % 10 == 0 and is_main_process:
                logger.info(f'Running through training batches {batchIx} of {maxBatches}. Input Size {ecg.shape}')

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                ecg = ecg[:,leads,:].to(device)
                clinicalParam = clincalParam.to(device).unsqueeze(1)

                predictedVal = network(ecg)
                lossVal = lossFun(predictedVal, clinicalParam, lossParams)
                lossVal.backward()
                optimizer.step()
                running_loss = running_loss + lossVal.item()
        
        # Calculate training loss - need to account for DDP
        if ddp_enabled and dist.is_initialized():
            # Gather loss from all processes
            loss_tensor = torch.tensor(running_loss, device=device)
            gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, loss_tensor)
            total_loss = sum([l.item() for l in gathered_losses])
            # Total dataset size (same across all processes)
            total_samples = len(trainDataLoader_normals.dataset) + len(trainDataLoader_abnormals.dataset)
            currTrainingLoss = total_loss / total_samples
        else:
            currTrainingLoss = running_loss / (len(trainDataLoader_normals.dataset) + len(trainDataLoader_abnormals.dataset))

        if is_main_process:
            logger.info(f'Using lead {leads}. Running through training batches end of {maxBatches}')
            logger.info(f"Epoch {ep+1} train loss {currTrainingLoss}, Diff {currTrainingLoss - prevTrainingLoss}")
        prevTrainingLoss = currTrainingLoss
        
        if is_main_process:
            logger.info('Evaling test')
        currTestLoss, allParams_test, allPredictions_test, ecgFig = evaluate_balanced(
            network, [testDataLoader], lossFun, lossParams, leads, device, ddp_enabled=ddp_enabled, lookForFig=False)
        
        if is_main_process:
            logger.info('Evaling train')
        currTrainLoss, allParams_train, allPredictions_train, _ = evaluate_balanced(
            network, [trainDataLoader_normals, trainDataLoader_abnormals], lossFun, lossParams, leads, device, ddp_enabled=ddp_enabled)
        
        if is_main_process:
            logger.info(f"train loss: {currTrainLoss}, val loss: {currTestLoss}")

        # Process results - only on main process since we gathered all data there
        if is_main_process:
            allParams_train = allParams_train.clone().detach().cpu().numpy()
            allPredictions_train = allPredictions_train.clone().detach().cpu().numpy()
            allParams_test = allParams_test.clone().detach().cpu().numpy()
            allPredictions_test = allPredictions_test.clone().detach().cpu().numpy()

            # Labels are already binary (0 or 1) from the dataset, no conversion needed
            allParams_train = allParams_train.astype(float)
            allParams_test = allParams_test.astype(float)

            #get roc curve and auc
            #print('Calculating ROC and other metrics')
            falsePos_train, truePos_train, thresholds_train = metrics.roc_curve(allParams_train,allPredictions_train)
            falsePos_test, truePos_test, thresholds_test = metrics.roc_curve(allParams_test,allPredictions_test)
			
            evalMetric_train = metrics.roc_auc_score(allParams_train, allPredictions_train)
            evalMetric_test = metrics.roc_auc_score(allParams_test, allPredictions_test)
        else:
            # Non-main processes don't compute metrics, but we need to sync
            evalMetric_test = 0.0
            evalMetric_train = 0.0
            falsePos_train = None
            truePos_train = None
            falsePos_test = None
            truePos_test = None
        
        # Sync best metric across processes if DDP
        if ddp_enabled and dist.is_initialized():
            eval_metric_tensor = torch.tensor(evalMetric_test, device=device)
            dist.broadcast(eval_metric_tensor, src=0)
            evalMetric_test = eval_metric_tensor.item()

        if is_main_process and evalMetric_test > bestEvalMetric_test:
             bestEvalMetric_test = evalMetric_test

             best_model = copy.deepcopy(model_for_saving.state_dict())
             model_save_path = os.path.join(modelSaveDir, modelName)
             os.makedirs(model_save_path, exist_ok=True)
             torch.save(best_model, os.path.join(model_save_path, f"{modelName}_best.pth"))
             logger.info(f"Model saved at {os.path.join(model_save_path, f'{modelName}_best.pth')} @ Epoch {ep+1} of {numEpoch}")
        
        # Sync bestEvalMetric_test across processes
        if ddp_enabled and dist.is_initialized():
            best_metric_tensor = torch.tensor(bestEvalMetric_test, device=device)
            dist.broadcast(best_metric_tensor, src=0)
            bestEvalMetric_test = best_metric_tensor.item()

        
        # precision, recall, thresholds = metrics.precision_recall_curve(allParams_test, allPredictions_test)
        # denominator = recall+precision
        # if np.any(np.isclose(denominator,[0.0])):
        #     print('\nSome precision+recall were zero. Setting to 1.\n')
        #     denominator[np.isclose(denominator,[0.0])] = 1

        # f1_scores = 2*recall*precision/(recall+precision)
        # f1_scores[np.isnan(f1_scores)] = 0
        # maxIx = np.argmax(f1_scores)

        # f1_score_test_max = f1_scores[maxIx]
        # thresholdForMax = thresholds[maxIx]

        # acc_test_f1max = metrics.balanced_accuracy_score(allParams_test,(allPredictions_test>thresholdForMax).astype('float'))
        # acc_train_f1max = metrics.balanced_accuracy_score(allParams_train,(allPredictions_train>thresholdForMax).astype('float'))

        # acc_test = metrics.balanced_accuracy_score(allParams_test,(allPredictions_test>0.5).astype('float'))
        # acc_train = metrics.balanced_accuracy_score(allParams_train,(allPredictions_train>0.5).astype('float'))
        
        # if acc_test > best_acc:
        #     best_acc = acc_test
        
        # print(f'Weighted Acc at 50% cutoff: {acc_train:.4f} train {acc_test:.4f} test')

        if is_main_process:
            logger.info(f'train score: {evalMetric_train:0.4f} test score: {evalMetric_test:0.4f}')
            logger.info(f'best AUC Test: {bestEvalMetric_test:0.4f}')

            if logToWandB and WANDB_AVAILABLE:
                logger.info('Logging to wandb')
                plt.figure(1)
                fig,ax1 = plt.subplots(1,2)

                ax1[0].plot(falsePos_train,truePos_train)
                ax1[0].set_title(f'ROC train, AUC: {evalMetric_train:0.3f}')
                ax1[1].plot(falsePos_test,truePos_test)
                ax1[1].set_title(f'ROC test, AUC: {evalMetric_test:0.3f}')
                plt.suptitle(f'ROC curves train AUC: {evalMetric_train:0.3f} test AUC: {evalMetric_test:0.3f}')

                logDict = {
                     'Epoch': ep,
                     'Training Loss': currTrainingLoss,
                     'Test Loss': currTestLoss,
                     'auc test': evalMetric_test,
                     'auc train': evalMetric_train,
                     'ROCs individual': wandb.Image(fig),
                }
                wandb.log(logDict)
                plt.close(fig)
            
            if evalMetric_train >= 0.999:
                logger.info('Training AUC is 1.0')
                exhausted += 1
                if exhausted == 3:
                    logger.info(f'Early stopping @ epoch {ep+1} with best AUC test {bestEvalMetric_test}')
                    break
        
        # Sync early stopping decision across processes
        if ddp_enabled and dist.is_initialized():
            exhausted_tensor = torch.tensor(exhausted, device=device)
            dist.broadcast(exhausted_tensor, src=0)
            exhausted = exhausted_tensor.item()
            if exhausted >= 3:
                break

    if is_main_process:
        logger.info(f"Best AUC Test: {bestEvalMetric_test}")
    return bestEvalMetric_test