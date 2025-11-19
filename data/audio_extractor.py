import torch
import torchaudio
import pandas as pd
import time
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
model.to(device)
model.eval()


class AudioDataset(Dataset):
    def __init__(self, dataframe, processor, max_length=250000):
        self.dataframe = dataframe
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio_path = self.dataframe.iloc[idx]['path']
        hash_id = self.dataframe.iloc[idx]['hash_id']
        emotion = self.dataframe.iloc[idx]['annotator_emo']

        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)

            waveform = waveform.squeeze()

            if waveform.size(0) > self.max_length:
                waveform = waveform[:self.max_length]

            inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt",
                                    padding="max_length", max_length=self.max_length)
            input_values = inputs.input_values.squeeze(0)

            return {
                'input_values': input_values,
                'hash_id': hash_id,
                'emotion': emotion
            }

        except Exception as e:
            print(f"Error processing file {audio_path}: {e}")
            return {
                'input_values': torch.zeros(self.max_length),
                'hash_id': hash_id,
                'emotion': emotion
            }


@torch.no_grad()
def extract_features_batch(batch_input_values):
    batch_input_values = batch_input_values.to(device)
    outputs = model(batch_input_values)
    features = outputs.last_hidden_state.mean(dim=1)
    return features.cpu().numpy()


def collate_fn(batch):
    input_values = torch.stack([item['input_values'] for item in batch])
    hash_ids = [item['hash_id'] for item in batch]
    emotions = [item['emotion'] for item in batch]

    return {
        'input_values': input_values,
        'hash_ids': hash_ids,
        'emotions': emotions
    }


def extract_audio_features(csv_path, base_path, output_path, batch_size=8, num_workers=4):
    print("Reading data...")
    data = pd.read_csv(csv_path)
    data_full_path = data.copy()
    data_full_path['path'] = base_path + data_full_path['audio_path']

    dataset = AudioDataset(data_full_path, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print("Starting feature extraction...")
    start_time = time.time()

    feature_list = []
    hash_id_list = []
    emotion_list = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        batch_input_values = batch['input_values']
        batch_hash_ids = batch['hash_ids']
        batch_emotions = batch['emotions']

        batch_features = extract_features_batch(batch_input_values)

        feature_list.extend(batch_features)
        hash_id_list.extend(batch_hash_ids)
        emotion_list.extend(batch_emotions)

    print("Creating DataFrame and saving results...")
    feature_df = pd.DataFrame(feature_list)
    feature_df['name'] = hash_id_list
    feature_df['emotion'] = emotion_list

    feature_df.to_csv(output_path, index=False)

    end_time = time.time() - start_time
    print(f"Results saved to: {output_path}")
    print(f"Feature extraction completed in: {end_time:.2f} seconds")

    return end_time


def main():
    csv_path = '/root/Downloads/test_pre_crowd.csv'
    base_path = '/root/Downloads/crowd/crowd_test/'
    output_path = "/root/maidung/wav2vec2_test_audio_features_unique_new.csv"

    extract_audio_features(csv_path, base_path, output_path)


if __name__ == "__main__":
    main()
