#!/usr/bin/env python3
"""
Train TimeSformer-based binary classification model with AMP & head-only fine-tuning
on 30-frame clips in a single-GPU VM (Standard_NC96ads_A100_v4).

Key memory-savers:
  - Mixed precision (AMP)
  - Gradient checkpointing
  - Freeze all backbone weights; train only the final classifier head
  - batch_size=1, num_workers=0
  - spawn + file_system sharing to avoid SIGBUS
"""

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch.multiprocessing as tmp
tmp.set_sharing_strategy('file_system')

import multiprocessing.resource_tracker as rt
_orig_register, _orig_unregister = rt.register, rt.unregister
def _patched_register(name, rtype):
    if rtype == "semaphore": return
    return _orig_register(name, rtype)
def _patched_unregister(name, rtype):
    if rtype == "semaphore": return
    try: return _orig_unregister(name, rtype)
    except KeyError: return
rt.register, rt.unregister = _patched_register, _patched_unregister

import argparse, os, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from PIL import Image
from transformers import (
    TimesformerConfig,
    TimesformerForVideoClassification,
    AutoImageProcessor
)
from datasets.raster_dataset import DatasetSequence2DBinary
from utils.download import download_from_blob
from utils.log_config import get_custom_logger

logger = get_custom_logger('train_timesformer_binary')

def train_model(train_loader, val_loader, device, processor,
                model, criterion, optimizer, num_epochs, patience):
    scaler = GradScaler()
    best_recall, epochs_no_improve, best_epoch = 0.0, 0, 0
    best_metrics, best_dir = {}, None

    mlflow.start_run()
    start_time = time.time()

    for epoch in range(1, num_epochs+1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        model.train()
        train_loss = train_correct = train_total = 0

        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            B, L, C, H, W = seq.shape

            # build uint8 RGB frames
            frames = (seq.cpu()
                      .repeat(1,1,3,1,1)
                      .permute(0,1,3,4,2)
                      .numpy())
            frames = np.clip(frames,0,255).astype(np.uint8)
            videos = [[Image.fromarray(frames[b,i]) for i in range(L)] for b in range(B)]
            inputs = processor(videos, return_tensors="pt").to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            logits = outputs.logits
            train_loss += loss.item()*B
            preds = logits.argmax(dim=-1)
            train_correct += (preds==labels).sum().item()
            train_total += B

        avg_train_loss = train_loss/train_total
        train_acc = train_correct/train_total
        mlflow.log_metric('train_loss', avg_train_loss, step=epoch)
        mlflow.log_metric('train_accuracy', train_acc, step=epoch)
        logger.info(f"Train loss={avg_train_loss:.4f}, acc={train_acc:.2f}")

        # validation
        model.eval()
        val_loss=val_correct=val_total=0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.to(device), labels.to(device)
                B, L, C, H, W = seq.shape
                frames = (seq.cpu()
                          .repeat(1,1,3,1,1)
                          .permute(0,1,3,4,2)
                          .numpy())
                frames = np.clip(frames,0,255).astype(np.uint8)
                videos = [[Image.fromarray(frames[b,i]) for i in range(L)] for b in range(B)]
                inputs = processor(videos, return_tensors="pt").to(device)

                with autocast():
                    outputs = model(**inputs, labels=labels)
                    loss, logits = outputs.loss, outputs.logits

                val_loss += loss.item()*B
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=-1)
                val_correct += (preds==labels).sum().item()
                val_total += B

                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        avg_val_loss = val_loss/val_total
        val_acc = val_correct/val_total
        preds = np.argmax(all_probs,axis=1)
        recall = recall_score(all_labels,preds,average='binary',pos_label=1,zero_division=0)
        precision = precision_score(all_labels,preds,average='binary',pos_label=1,zero_division=0)
        f1 = f1_score(all_labels,preds,average='binary',pos_label=1,zero_division=0)
        try:
            fpr,tpr,_ = roc_curve(all_labels,[p[1] for p in all_probs],pos_label=1)
            auc_score = auc(fpr,tpr)
        except:
            auc_score = 0.0

        mlflow.log_metrics({
            'val_loss':avg_val_loss,
            'val_accuracy':val_acc,
            'val_recall':recall,
            'val_precision':precision,
            'val_f1':f1,
            'val_auc':auc_score
        },step=epoch)
        logger.info(f"Val loss={avg_val_loss:.4f}, acc={val_acc:.2f}, rec={recall:.2f}")

        # early stop & save
        if recall > best_recall:
            best_recall, best_epoch = recall, epoch
            best_metrics = dict(loss=avg_val_loss, acc=val_acc,
                                precision=precision, f1=f1, auc=auc_score)
            best_dir = f"outputs/timesformer_best_ep{epoch}"
            os.makedirs(best_dir,exist_ok=True)
            model.save_pretrained(best_dir)
            cm = confusion_matrix(all_labels,preds)
            disp=ConfusionMatrixDisplay(cm); disp.plot()
            plt.savefig(os.path.join(best_dir,'confmat.png')); plt.close()
            epochs_no_improve=0
        else:
            epochs_no_improve+=1
            if epochs_no_improve>=patience:
                logger.info("Early stopping") 
                break

    elapsed=time.time()-start_time
    logger.info(f"Done in {elapsed:.1f}s; best recall {best_recall:.3f}@ep{best_epoch}")
    final_metrics = {
        "best_epoch":best_epoch,
        **{"best_"+k:v for k,v in best_metrics.items()}
    }
    mlflow.log_metrics(final_metrics,step=best_epoch)
    mlflow.end_run()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train_dir',required=True)
    p.add_argument('--val_dir',required=True)
    p.add_argument('--sequence_length',type=int,default=30)
    p.add_argument('--batch_size',type=int,default=1)
    p.add_argument('--num_epochs',type=int,default=20)
    p.add_argument('--learning_rate',type=float,default=5e-5)
    p.add_argument('--model_name_or_path',default='facebook/timesformer-base-finetuned-k400')
    p.add_argument('--patience',type=int,default=3)
    args=p.parse_args()

    load_dotenv()
    acc, key, cont = (os.getenv(x) for x in 
        ("AZURE_STORAGE_ACCOUNT","AZURE_STORAGE_KEY","BLOB_CONTAINER"))
    logger.info("Downloading blobs…")
    download_from_blob(acc,key,cont,args.train_dir)
    download_from_blob(acc,key,cont,args.val_dir)

    logger.info("Building datasets…")
    train_ds = DatasetSequence2DBinary(args.train_dir,args.sequence_length)
    val_ds   = DatasetSequence2DBinary(args.val_dir,args.sequence_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False,num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TimesformerConfig.from_pretrained(
        args.model_name_or_path,num_labels=2,num_frames=args.sequence_length)
    config.attention_type="joint_space_time"
    model = TimesformerForVideoClassification.from_pretrained(
        args.model_name_or_path,config=config,ignore_mismatched_sizes=True
    ).to(device)

    # freeze backbone, train only classifier head
    for n,p in model.named_parameters():
        if "classifier" not in n:
            p.requires_grad=False

    model.gradient_checkpointing_enable()
    processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p:p.requires_grad,model.parameters()),
        lr=args.learning_rate,weight_decay=1e-5
    )

    train_model(train_loader,val_loader,device,processor,
                model,criterion,optimizer,
                num_epochs=args.num_epochs,patience=args.patience)

if __name__=='__main__':
    main()
