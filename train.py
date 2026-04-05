import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import (
    DEVICE, EPOCHS, LEARNING_RATE, BATCH_SIZE,
    MODELS_DIR, BINARY_CLASSES, MULTI_CLASSES
)
from data_loader import (
    load_fakemusiccaps, load_combined_dataset, load_custom_dataset,
    create_data_loaders
)
from model import get_model


class Trainer:
    """Training class for Vocaluity models."""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 num_classes=2, class_names=None, class_weights=None):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.num_classes  = num_classes
        self.class_names  = class_names or [str(i) for i in range(num_classes)]

        # Weighted loss to handle class imbalance
        weight_tensor = class_weights.to(DEVICE) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Adam with weight decay (L2 regularisation) to reduce overfitting
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )

        # Reduce LR when validation loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        self.history = {
            'train_loss': [],
            'val_loss':   [],
            'train_acc':  [],
            'val_acc':    []
        }

        self.best_val_acc   = 0.0
        self.best_model_path = None

    def train_epoch(self):
        """Train for one epoch and return (loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        all_preds  = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc='Training')
        for features, labels in pbar:
            features = features.to(DEVICE, non_blocking=True)
            labels   = labels.to(DEVICE, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss    = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc  = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc

    def validate(self, loader=None):
        """Evaluate on a DataLoader and return metrics + raw predictions."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        running_loss = 0.0
        all_preds  = []
        all_labels = []
        all_probs  = []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(DEVICE, non_blocking=True)
                labels   = labels.to(DEVICE, non_blocking=True)

                outputs = self.model(features)
                loss    = self.criterion(outputs, labels)

                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / len(loader)
        epoch_acc  = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

    def train(self, epochs=EPOCHS, save_best=True, early_stopping_patience=10):
        """
        Full training loop.

        Args:
            epochs:                    Maximum number of epochs
            save_best:                 Save checkpoint whenever val_acc improves
            early_stopping_patience:   Stop after this many epochs without
                                       improvement in val_loss (0 = disabled)
        """
        print(f"\nStarting training for up to {epochs} epochs...")
        print(f"Device:             {DEVICE}")
        print(f"Batch size:         {BATCH_SIZE}")
        print(f"Learning rate:      {LEARNING_RATE}")
        print(f"Weight decay:       1e-4")
        print(f"Early stop patience:{early_stopping_patience}")
        print("-" * 60)

        best_val_loss      = float('inf')
        early_stop_counter = 0

        for epoch in range(epochs):
            # --- Train ---
            train_loss, train_acc = self.train_epoch()

            # --- Validate ---
            val_loss, val_acc, _, _, _ = self.validate()

            # --- LR scheduler ---
            self.scheduler.step(val_loss)

            # --- Record history ---
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch [{epoch+1}/{epochs}]  "
                  f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
                  f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

            # --- Save best model (by val_acc) ---
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc    = val_acc
                self.best_model_path = self._save_model(epoch, val_acc)
                print(f"  > New best model saved! Val Acc: {val_acc:.4f}")

            # --- Early stopping (tracks val_loss to catch overfitting) ---
            if early_stopping_patience > 0:
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss      = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs "
                              f"({early_stopping_patience} epochs without val_loss improvement).")
                        break

        print("\nTraining complete!")
        return self.history

    def evaluate(self, loader=None):
        """Evaluate model on test set and print full metrics."""
        if loader is None:
            loader = self.test_loader

        print("\nEvaluating on test set...")
        _, test_acc, preds, labels, probs = self.validate(loader)

        metrics = {
            'accuracy':  accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall':    recall_score(labels, preds, average='weighted', zero_division=0),
            'f1':        f1_score(labels, preds, average='weighted', zero_division=0)
        }

        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")

        all_labels_list = list(range(self.num_classes))
        print("\nClassification Report:")
        print(classification_report(
            labels, preds,
            labels=all_labels_list,
            target_names=self.class_names,
            zero_division=0
        ))

        cm = confusion_matrix(labels, preds, labels=all_labels_list)
        print("\nConfusion Matrix:")
        print(cm)

        return metrics, preds, labels, probs

    def _save_model(self, epoch, val_acc):
        """Save model checkpoint."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"vocaluity_epoch{epoch+1}_acc{val_acc:.4f}_{timestamp}.pth"
        filepath  = MODELS_DIR / filename
        torch.save({
            'epoch':              epoch,
            'model_state_dict':   self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc':            val_acc,
            'history':            self.history
        }, filepath)
        return filepath

    def load_model(self, filepath):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded model from {filepath}")
        return checkpoint.get('val_acc', 0)

    def plot_history(self, save_path=None):
        """Plot training / validation loss and accuracy curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self.history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(self.history['val_loss'],   label='Val Loss',   marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history['train_acc'], label='Train Acc', marker='o')
        ax2.plot(self.history['val_acc'],   label='Val Acc',   marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        plt.show()
        return fig

    def plot_confusion_matrix(self, labels, preds, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(labels, preds)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.show()
        return fig


def main():
    print("=" * 60)
    print("VOCALUITY - AI Vocal Detection Training")
    print("=" * 60)


    USE_BOTH_DATASETS = True # Set to True to train on combined dataset (binary or multi-class)
    # USE_BOTH_DATASETS = False  # Set to False to train on FakeMusicCaps only (multi-class)

    BINARY_CLASSIFICATION = True # Set to True for binary classification (real vs AI)
    # BINARY_CLASSIFICATION = False # Set to False for multi-class classification (real + 4 AI generators)

    # Model architecture: 'simple', 'resnet', or 'lightweight'
    MODEL_TYPE = 'simple'

    # Load dataset
    if USE_BOTH_DATASETS:
        print("\nLoading combined FakeMusicCaps + MusicCaps dataset...")
        try:
            file_paths, labels, label_map = load_combined_dataset(
                binary=BINARY_CLASSIFICATION
            )
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            return
    else:
        print("\nLoading FakeMusicCaps dataset only...")
        try:
            file_paths, labels, label_map = load_fakemusiccaps(
                binary=BINARY_CLASSIFICATION
            )
        except FileNotFoundError as e:
            print(f"\n{e}")
            return

    # Crete data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        file_paths, labels,
        batch_size=BATCH_SIZE
    )

    # Model creation
    num_classes  = len(label_map)
    class_names  = list(label_map.keys())

    print(f"\nCreating '{MODEL_TYPE}' model with {num_classes} classes: {class_names}")
    model = get_model(MODEL_TYPE, num_classes=num_classes)

    # Train
    trainer = Trainer(
        model, train_loader, val_loader, test_loader,
        num_classes=num_classes,
        class_names=class_names,
        class_weights=class_weights   # weighted loss for class imbalance
    )

    history = trainer.train(
        epochs=EPOCHS,
        early_stopping_patience=10    # stop if val_loss stalls for 10 epochs
    )

    # Evaluate on test set using best model (reload weights from best checkpoint)
    if trainer.best_model_path:
        print(f"\nReloading best model weights from: {trainer.best_model_path.name}")
        trainer.load_model(trainer.best_model_path)

    metrics, preds, labels, probs = trainer.evaluate()

    # Save results and plots
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    trainer.plot_history(save_path=results_dir / "training_history.png")
    trainer.plot_confusion_matrix(labels, preds,
                                  save_path=results_dir / "confusion_matrix.png")

    with open(results_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    print(f"Best model:  {trainer.best_model_path}")

    return trainer, metrics


if __name__ == "__main__":
    result = main()
    if result is not None:
        trainer, metrics = result
