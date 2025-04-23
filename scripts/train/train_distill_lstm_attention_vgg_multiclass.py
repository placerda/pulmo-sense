import argparse
import os
import time
import traceback
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torch.utils.data import DataLoader, Subset

from dotenv import load_dotenv

from datasets import DatasetSequence2D
from utils.download import download_from_blob, download_from_blob_with_access_key
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('train_distill_lstm_attention_vgg_multiclass')

# ----------------------------------------------------------------------
# 1) Teacher Model: Pretrained VGG
# ----------------------------------------------------------------------
class TeacherVGG(nn.Module):
    """
    This is your 'teacher' model: a fully trained VGG that directly outputs class logits.
    """
    def __init__(self, vgg_weights_path, num_classes=3):
        super(TeacherVGG, self).__init__()
        self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)

        # Replace last layer with your classifier shape
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

        # Load the fully fine-tuned weights
        state_dict = torch.load(vgg_weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # x is [batch_size, 1, H, W]; convert to 3-ch
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)  # [batch_size, num_classes]

# ----------------------------------------------------------------------
# 2) Student Feature Extractor (Partial VGG) or 2D CNN?
#    But here we want your existing LSTM+Attention with VGG-based features?
# ----------------------------------------------------------------------
class VGGExtractorNoFC(nn.Module):
    """
    A VGG that ends with the penultimate 4096-dim layer. 
    We'll freeze it. The Student LSTM+Attention reads those 4096 features.
    """
    def __init__(self, vgg_weights_path):
        super(VGGExtractorNoFC, self).__init__()
        base_vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        in_features = base_vgg.classifier[6].in_features
        base_vgg.classifier[6] = nn.Identity()
        # Load
        state_dict = torch.load(vgg_weights_path, map_location='cpu')
        base_vgg.load_state_dict(state_dict, strict=False)
        self.vgg = base_vgg
        self.embedding_dim = in_features

    def forward(self, x):
        # x [batch, 1, H, W]
        x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.vgg(x)  # [batch, 4096]

# ----------------------------------------------------------------------
# 3) LSTM + Attention Student
# ----------------------------------------------------------------------
class SimpleAttention(nn.Module):
    """
    Single-head additive attention for LSTM outputs.
    """
    def __init__(self, hidden_dim):
        super(SimpleAttention, self).__init__()
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        # lstm_outputs: [batch, seq, hidden_dim]
        energies = self.attn_score(lstm_outputs).squeeze(-1)  # [batch, seq]
        attn_weights = torch.softmax(energies, dim=1)
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        context = torch.sum(lstm_outputs * attn_weights_expanded, dim=1)
        return context, attn_weights

class LSTM_AttnStudent(nn.Module):
    """
    The "student" model: it sees sequence of VGG-extracted embeddings,
    runs them through LSTM+Attention, and outputs class logits.
    """
    def __init__(self, input_size=4096, hidden_size=128, num_layers=1, num_classes=3, dropout=0.5):
        super(LSTM_AttnStudent, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = SimpleAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: [batch, seq, 4096]
        """
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden_size]
        context, attn_weights = self.attention(lstm_out)  # [batch, hidden_size], [batch, seq]
        context = self.dropout(context)
        logits = self.fc(context)
        return logits, attn_weights

# ----------------------------------------------------------------------
# 4) Distillation Loss
# ----------------------------------------------------------------------
def distillation_loss(student_logits, teacher_logits, true_labels, alpha=0.5, T=2.0):
    """
    A common KD approach: combine KL-divergence between softmax outputs
    with standard cross-entropy.
      - student_logits: [batch, num_classes]
      - teacher_logits: [batch, num_classes]
      - true_labels: [batch] (class indices)
      - alpha: weighting between KD loss and CE
      - T: temperature
    """
    # 1) Hard label CE
    ce_loss = nn.CrossEntropyLoss()(student_logits, true_labels)

    # 2) Soft label KD (teacher as target)
    # Teacher probabilities
    teacher_probs = torch.softmax(teacher_logits / T, dim=1)
    # Student log-probs
    student_log_probs = torch.log_softmax(student_logits / T, dim=1)

    kl_div = nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs) * (T * T)

    return alpha * kl_div + (1 - alpha) * ce_loss

# ----------------------------------------------------------------------
# 5) Training with Distillation
# ----------------------------------------------------------------------
def train_model(train_loader, val_loader, num_epochs, learning_rate,
                teacher_model_path, student_vgg_weights_path,
                alpha=0.5, temperature=2.0):
    """
    teacher_model_path: path to the fully trained VGG classification model (teacher)
    student_vgg_weights_path: path to VGG weights for the student's feature extractor
    alpha, temperature: distillation hyperparams
    """
    start_time = time.time()
    my_logger.info('Starting Distillation Training: LSTM+Attention Student, VGG Teacher')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_logger.info(f'Using device={device}')

    try:
        # Teacher: a fully trained VGG classifier
        teacher_model = TeacherVGG(teacher_model_path, num_classes=3).to(device)
        teacher_model.eval()  # Freeze teacher
        for p in teacher_model.parameters():
            p.requires_grad = False

        # Student:
        #  - VGG feature extractor with final layer = Identity
        #  - LSTM + Attention classification head
        feature_extractor = VGGExtractorNoFC(student_vgg_weights_path).to(device)
        feature_extractor.eval()
        for p in feature_extractor.parameters():
            p.requires_grad = False

        student = LSTM_AttnStudent(input_size=feature_extractor.embedding_dim,
                                   hidden_size=128, num_layers=1, num_classes=3, dropout=0.5).to(device)

        optimizer = optim.Adam(student.parameters(), lr=learning_rate, weight_decay=1e-5)

        best_recall = 0.0
        early_stopping_patience = 3
        epochs_without_improvement = 0

        # 6) Training Loop
        for epoch in range(num_epochs):
            my_logger.info(f'Epoch {epoch+1}/{num_epochs}')
            student.train()

            total_loss = 0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                bsz, seq_len, ch, h, w = inputs.size()

                # Flatten for teacher and feature_extractor
                flat_input = inputs.view(bsz * seq_len, ch, h, w)

                with torch.no_grad():
                    # Teacher predictions on raw slices (each slice is a single sample)
                    teacher_logits = teacher_model(flat_input)  # shape [bsz*seq_len, 3]

                # Student's VGG feature extraction
                with torch.no_grad():
                    feats = feature_extractor(flat_input)  # [bsz*seq_len, 4096]

                feats = feats.view(bsz, seq_len, -1)
                # Student forward pass
                student_logits, attn_weights = student(feats)  # [bsz, 3]

                # We need to reshape teacher_logits to [bsz, seq_len, 3] then average or combine
                # But your teacher sees each slice individually, while your student sees the entire sequence.
                # One approach: we can average teacher's slice logits to get one teacher "logit" per batch item.
                teacher_logits = teacher_logits.view(bsz, seq_len, -1)
                teacher_logits_mean = teacher_logits.mean(dim=1)  # [bsz, 3]

                # Distillation loss
                loss = distillation_loss(
                    student_logits,          # student
                    teacher_logits_mean,     # teacher
                    labels,                  # ground truth
                    alpha=alpha,
                    T=temperature
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * bsz
                preds = student_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if i % 5 == 0:
                    batch_acc = (preds == labels).float().mean().item()
                    my_logger.info(f'[Epoch {epoch+1}] Batch [{i+1}/{len(train_loader)}], '
                                   f'Loss={loss.item():.4f}, Acc={batch_acc:.4f}')
                    mlflow.log_metric('running_train_loss', loss.item(), step=epoch*len(train_loader)+i)
                    mlflow.log_metric('running_train_accuracy', batch_acc, step=epoch*len(train_loader)+i)

            train_loss = total_loss / total
            train_acc = correct / total
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('train_accuracy', train_acc, step=epoch)
            my_logger.info(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}')

            # Validation
            student.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_labels, all_probs = [], []

            with torch.no_grad():
                for j, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    bsz, seq_len, ch, h, w = inputs.size()

                    flat_input = inputs.view(bsz * seq_len, ch, h, w)

                    # Teacher. We don't need teacher for validation, unless you want "teacher forcing." 
                    # But typically we just measure student vs. ground truth.
                    # teacher_logits = teacher_model(flat_input)

                    feats = feature_extractor(flat_input)
                    feats = feats.view(bsz, seq_len, -1)
                    student_logits, attn_weights = student(feats)

                    # Standard CE to measure val loss
                    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
                    val_loss += ce_loss.item() * bsz

                    probs = torch.softmax(student_logits, dim=1)
                    preds = probs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            val_loss /= val_total
            val_acc = val_correct / val_total
            val_preds = np.argmax(all_probs, axis=1)

            val_recall = recall_score(all_labels, val_preds, average='macro', zero_division=0)
            val_precision = precision_score(all_labels, val_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, val_preds, average='macro', zero_division=0)
            try:
                val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
            except ValueError:
                val_auc = 0.0

            my_logger.info(f'Val epoch={epoch+1}: loss={val_loss:.4f}, acc={val_acc:.4f}, '
                           f'recall={val_recall:.2f}, precision={val_precision:.2f}, f1={val_f1:.2f}, auc={val_auc:.2f}')
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_accuracy', val_acc, step=epoch)
            mlflow.log_metric('val_recall', val_recall, step=epoch)
            mlflow.log_metric('val_precision', val_precision, step=epoch)
            mlflow.log_metric('val_f1_score', val_f1, step=epoch)
            mlflow.log_metric('val_auc', val_auc, step=epoch)

            # Early Stopping
            if val_recall > best_recall:
                best_recall = val_recall
                epochs_without_improvement = 0

                output_dir = './outputs'
                os.makedirs(output_dir, exist_ok=True)
                file_prefix = f'distill_student_{val_total}val_{epoch+1:03}ep_{learning_rate:.5f}lr_{val_recall:.3f}rec'
                model_path = os.path.join(output_dir, f'{file_prefix}.pth')
                torch.save(student.state_dict(), model_path)
                my_logger.info(f"New best student model with recall={val_recall:.3f}")

                cm = confusion_matrix(all_labels, val_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap=plt.cm.Blues)
                plt.savefig(os.path.join(output_dir, f'{file_prefix}_confmat.png'))
                plt.close()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= 3:
                my_logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        my_logger.info("Finished Distillation Training.")
        my_logger.info(f"Total time: {time.time() - start_time:.2f}s")

    except Exception as e:
        my_logger.error(f"Error: {e}")
        my_logger.error(traceback.format_exc())

# ----------------------------------------------------------------------
# 6) Main
# ----------------------------------------------------------------------
def main():
    my_logger.info(f"Torch version: {torch.__version__}")
    mlflow.start_run()
    my_logger.info("MLflow run started")

    parser = argparse.ArgumentParser(description='Distill knowledge from a fully trained VGG teacher into LSTM+Attn student.')
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='ccccii')
    parser.add_argument("--run_cloud", action='store_true')
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--teacher_model_path", type=str, default='models/vgg_teacher.pth',
                        help="Path to fully-trained VGG classifier (teacher).")
    parser.add_argument("--student_vgg_weights_path", type=str, default='models/vgg_student_init.pth',
                        help="Path to VGG weights for student's feature extractor.")
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for KD vs CE.")
    parser.add_argument("--temperature", type=float, default=2.0, help="KD temperature")

    args = parser.parse_args()
    my_logger.info(f"Args: {args}")
    my_logger.info(f"Current Working Dir: {os.getcwd()}")

    if not args.run_cloud:
        dataset_folder = f"data/{args.dataset}"
    else:
        my_logger.info("Running in cloud mode, attempting to download dataset and pretrained model from blob storage")
        load_dotenv()
        storage_account = os.getenv("AZURE_STORAGE_ACCOUNT")
        storage_account_key = os.getenv("AZURE_STORAGE_KEY")
        container_name = os.getenv("BLOB_CONTAINER")

        # Download pretrained VGG if needed
        if not os.path.exists(args.teacher_model_path):
            model_uri = os.getenv('PRETRAINED_VGG_MODEL_URI', '')
            my_logger.info(f"Downloading pretrained VGG model from: {model_uri}")
            os.makedirs(os.path.dirname(args.teacher_model_path), exist_ok=True)
            download_from_blob_with_access_key(model_uri, storage_account_key, args.teacher_model_path)
            my_logger.info(f"VGG model downloaded to {args.teacher_model_path}")

        if not os.path.exists(args.student_vgg_weights_path):
            model_uri = os.getenv('PRETRAINED_VGG_MODEL_URI', '')
            my_logger.info(f"Downloading pretrained VGG model from: {model_uri}")
            os.makedirs(os.path.dirname(args.student_vgg_weights_path), exist_ok=True)
            download_from_blob_with_access_key(model_uri, storage_account_key, args.student_vgg_weights_path)
            my_logger.info(f"VGG model downloaded to {args.student_vgg_weights_path}")

        # Download dataset
        dataset_folder = args.dataset
        my_logger.info(f"Downloading dataset from blob storage: {dataset_folder}")
        download_from_blob(storage_account, storage_account_key, container_name, dataset_folder)


    # load dataset
    dataset = DatasetSequence2D(dataset_folder, sequence_length=args.sequence_length, max_samples=args.max_samples)
    labels = dataset.labels
    if args.i < 0 or args.i >= args.k:
        raise ValueError(f"Fold index i={args.i} invalid for k={args.k} folds")

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(dataset)), labels))
    train_idx, val_idx = splits[args.i]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    my_logger.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        teacher_model_path=args.teacher_model_path,
        student_vgg_weights_path=args.student_vgg_weights_path,
        alpha=args.alpha,
        temperature=args.temperature
    )

    mlflow.end_run()

if __name__ == "__main__":
    main()
