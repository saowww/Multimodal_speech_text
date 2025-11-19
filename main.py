import torch
import matplotlib.pyplot as plt
import time
import pickle
import os
from utils.data_loader import load_data_speech
from utils.train import train_model
from utils.plot import plot_training_history
from utils.utils import count_parameters
from model.mamba_model import MambaEmotionClassifier
from model.transformer_model import SpeechTextTransformer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def train_models(use_wandb=True, output_dir="./outputs"):
    os.makedirs(output_dir, exist_ok=True)

    train_loader, test_loader, class_names = load_data_speech(batch=128)

    input_dim = 768
    num_classes = len(class_names)

    results = {}

    print("=" * 60)
    print("Training Transformer Model")
    print("=" * 60)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="multimodal-emotion-classification",
            name="transformer-model",
            config={
                "architecture": "Transformer",
                "dataset": "emotion-multimodal",
                "epochs": 125,
                "batch_size": 128,
                "learning_rate": 0.001,
                "input_dim": input_dim,
                "num_classes": num_classes,
            }
        )
        wandb.config.update({"class_names": class_names.tolist()})

    hidden_dim = 512
    num_heads = 8
    num_layers = 3

    transformer_model = SpeechTextTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_classes=num_classes,
        num_layers=num_layers
    )

    total_params, trainable_params = count_parameters(transformer_model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.config.update({
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "total_params": total_params,
            "trainable_params": trainable_params
        })

    start_time = time.time()

    transformer_model, transformer_history, transformer_probs = train_model(
        transformer_model, train_loader, test_loader,
        num_epochs=125, lr=0.001, classes=class_names,
        model_name="transformer", use_wandb=use_wandb, output_dir=output_dir)

    end_time = time.time()
    transformer_time = end_time - start_time

    if use_wandb and WANDB_AVAILABLE:
        wandb.run.summary["total_execution_time"] = transformer_time
        wandb.finish()

    results['transformer'] = {
        'history': transformer_history,
        'final_probs': transformer_probs,
        'class_names': class_names,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_time': transformer_time
    }

    print("\n" + "=" * 60)
    print("Training Mamba Model")
    print("=" * 60)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="multimodal-emotion-classification",
            name="mamba-model",
            config={
                "architecture": "Mamba",
                "dataset": "emotion-multimodal",
                "epochs": 125,
                "batch_size": 128,
                "learning_rate": 0.001,
                "input_dim": input_dim,
                "num_classes": num_classes,
            }
        )
        wandb.config.update({"class_names": class_names.tolist()})

    hidden_dim = 192
    d_state = 32
    dropout_rate = 0.3

    mamba_model = MambaEmotionClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        d_state=d_state,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )

    total_params, trainable_params = count_parameters(mamba_model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.config.update({
            "hidden_dim": hidden_dim,
            "d_state": d_state,
            "dropout_rate": dropout_rate,
            "total_params": total_params,
            "trainable_params": trainable_params
        })

    start_time = time.time()

    mamba_model, mamba_history, mamba_probs = train_model(
        mamba_model, train_loader, test_loader,
        num_epochs=125, lr=0.001, classes=class_names,
        model_name="mamba", use_wandb=use_wandb, output_dir=output_dir)

    end_time = time.time()
    mamba_time = end_time - start_time

    if use_wandb and WANDB_AVAILABLE:
        wandb.run.summary["total_execution_time"] = mamba_time
        wandb.finish()

    results['mamba'] = {
        'history': mamba_history,
        'final_probs': mamba_probs,
        'class_names': class_names,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_time': mamba_time
    }

    total_time = transformer_time + mamba_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    results_path = os.path.join(output_dir, 'model_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print("Training completed and results saved!")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(transformer_history['epoch_times'])+1),
             transformer_history['epoch_times'], label='Transformer')
    plt.plot(range(1, len(mamba_history['epoch_times'])+1),
             mamba_history['epoch_times'], label='Mamba')
    plt.title('Training Time by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_training_history(transformer_history)
    plot_training_history(mamba_history)

    return results


if __name__ == "__main__":
    train_models(use_wandb=True, output_dir="./outputs")
