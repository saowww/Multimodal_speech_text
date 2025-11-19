import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils.plot import plot_confusion_matrix

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def train_model(model, train_loader, test_loader, num_epochs=50, lr=0.001, patience=18, classes=None,
                model_name="model", use_wandb=False, output_dir="./"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, threshold=1e-4, threshold_mode='abs')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'epoch_times': []
    }

    best_f1 = 0.0
    early_stopping_counter = 0
    min_lr = 1e-6
    best_model_state = None

    os.makedirs(output_dir, exist_ok=True)

    if use_wandb and WANDB_AVAILABLE:
        wandb.watch(model, log="all", log_freq=10)

    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        train_true = []
        train_pred = []

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_audio, batch_text, batch_labels in progress_bar:
            batch_audio, batch_text, batch_labels = batch_audio.to(
                device), batch_text.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_audio, batch_text)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_true.extend(batch_labels.cpu().numpy())
            train_pred.extend(predicted.cpu().numpy())

            progress_bar.set_postfix(
                {'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})

        train_acc = accuracy_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred, average='weighted')
        train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_true = []
        val_pred = []

        with torch.no_grad():
            progress_bar = tqdm(
                test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch_audio, batch_text, batch_labels in progress_bar:
                batch_audio, batch_text, batch_labels = batch_audio.to(
                    device), batch_text.to(device), batch_labels.to(device)

                outputs = model(batch_audio, batch_text)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_true.extend(batch_labels.cpu().numpy())
                val_pred.extend(predicted.cpu().numpy())

                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average='weighted')
        val_loss = val_loss / len(test_loader)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        history['epoch_times'].append(epoch_time)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        current_lr = optimizer.param_groups[0]['lr']

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "learning_rate": current_lr,
                "epoch_time": epoch_time
            })

        if val_f1 > best_f1:
            best_f1 = val_f1
            early_stopping_counter = 0
            best_model_state = model.state_dict().copy()

            model_path = os.path.join(
                output_dir, f'best_{model_name}_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_f1': best_f1
            }, model_path)

            improvement = int(val_f1 * 1000)
            checkpoint_path = os.path.join(
                output_dir, f'checkpoint_{model_name}_f1_{improvement}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history,
                'f1_score': val_f1
            }, checkpoint_path)

            print(f"New best model saved with F1: {best_f1:.4f}")

            if use_wandb and WANDB_AVAILABLE:
                wandb.run.summary["best_f1"] = best_f1
                wandb.run.summary["best_epoch"] = epoch + 1
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} epochs")

        scheduler.step(val_loss)

        if current_lr <= min_lr and early_stopping_counter >= patience:
            print(
                f"Early stopping at epoch {epoch+1} after learning rate reduction")
            break

        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print("-" * 50)

    total_end_time = time.time()
    total_train_time = total_end_time - total_start_time

    hours, remainder = divmod(total_train_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Average time per epoch: {np.mean(history['epoch_times']):.2f}s")

    if use_wandb and WANDB_AVAILABLE:
        wandb.run.summary["total_train_time"] = total_train_time
        wandb.run.summary["avg_epoch_time"] = np.mean(history['epoch_times'])

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loading best model for final evaluation...")
    else:
        checkpoint_path = os.path.join(
            output_dir, f'best_{model_name}_model.pth')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    final_true = []
    final_pred = []
    final_probs = []

    with torch.no_grad():
        for batch_audio, batch_text, batch_labels in test_loader:
            batch_audio = batch_audio.to(device)
            batch_text = batch_text.to(device)

            outputs = model(batch_audio, batch_text)
            probabilities = F.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)
            final_true.extend(batch_labels.cpu().numpy())
            final_pred.extend(predicted.cpu().numpy())
            final_probs.extend(probabilities.cpu().numpy())

    final_acc = accuracy_score(final_true, final_pred)
    final_f1 = f1_score(final_true, final_pred, average='weighted')

    print("\n" + "="*60)
    print("Final model evaluation:")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"F1 Score: {final_f1:.4f}")
    print("="*60)

    if use_wandb and WANDB_AVAILABLE:
        wandb.run.summary["final_acc"] = final_acc
        wandb.run.summary["final_f1"] = final_f1

    results = {
        'history': history,
        'final_probs': final_probs,
        'final_true': final_true,
        'final_pred': final_pred,
        'class_names': classes,
        'final_acc': final_acc,
        'final_f1': final_f1,
        'training_time': total_train_time
    }

    results_path = os.path.join(output_dir, f'{model_name}_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    if use_wandb and WANDB_AVAILABLE:
        wandb.save(results_path)

    if classes is not None:
        cm = confusion_matrix(final_true, final_pred)
        cm_path = os.path.join(
            output_dir, f'{model_name}_confusion_matrix.png')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(cm_path)
        plt.close()

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({"confusion_matrix": wandb.Image(cm_path)})

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        norm_cm_path = os.path.join(
            output_dir, f'{model_name}_normalized_confusion_matrix.png')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(norm_cm_path)
        plt.close()

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({"normalized_confusion_matrix": wandb.Image(norm_cm_path)})

        plot_confusion_matrix(final_true, final_pred,
                              classes, title='Confusion Matrix')

    history_plot_path = os.path.join(
        output_dir, f'{model_name}_training_history.png')
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(history_plot_path)
    plt.close()

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"training_history": wandb.Image(history_plot_path)})

    epoch_times_path = os.path.join(
        output_dir, f'{model_name}_epoch_times.png')
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history['epoch_times'])+1), history['epoch_times'])
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(epoch_times_path)
    plt.close()

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"epoch_times": wandb.Image(epoch_times_path)})

    return model, history, final_probs
