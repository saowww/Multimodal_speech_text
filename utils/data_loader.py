import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def load_data_speech(batch=32):
    train_audio = pd.read_csv(
        '/kaggle/input/crowd-unique-features/wav2vec2_train_audio_features_unique_80.csv')
    test_audio = pd.read_csv(
        '/kaggle/input/crowd-unique-features/wav2vec2_test_audio_features_unique_20.csv')
    train_text = pd.read_csv(
        '/kaggle/input/crowd-unique-features/rubert_train_text_features_unique_80.csv')
    test_text = pd.read_csv(
        '/kaggle/input/crowd-unique-features/rubert_test_text_features_unique_20.csv')

    X_train_audio = train_audio.drop(['emotion', 'name'], axis=1).values
    X_train_text = train_text.drop(['emotion', 'name'], axis=1).values
    X_test_audio = test_audio.drop(['emotion', 'name'], axis=1).values
    X_test_text = test_text.drop(['emotion', 'name'], axis=1).values

    y_train = train_audio['emotion']
    y_test = test_audio['emotion']

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    X_train_audio = torch.tensor(X_train_audio, dtype=torch.float32)
    X_train_text = torch.tensor(X_train_text, dtype=torch.float32)
    X_test_audio = torch.tensor(X_test_audio, dtype=torch.float32)
    X_test_text = torch.tensor(X_test_text, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_audio, X_train_text, y_train)
    test_dataset = TensorDataset(X_test_audio, X_test_text, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    return train_loader, test_loader, label_encoder.classes_
