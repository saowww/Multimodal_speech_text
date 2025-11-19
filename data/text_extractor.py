from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
model.to(device)
model.eval()


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = str(self.dataframe.iloc[idx]['speaker_text'])
        hash_id = self.dataframe.iloc[idx]['hash_id']
        emotion = self.dataframe.iloc[idx]['speaker_emo']

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None
        )

        item = {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'hash_id': hash_id,
            'emotion': emotion
        }

        if 'token_type_ids' in inputs:
            item['token_type_ids'] = torch.tensor(
                inputs['token_type_ids'], dtype=torch.long)

        return item


@torch.no_grad()
def extract_features_batch(batch_inputs):
    batch_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_inputs.items()
                    if k not in ['hash_id', 'emotion']}

    outputs = model(**batch_inputs)
    features = outputs.last_hidden_state[:, 0, :]
    return features.cpu().numpy()


def extract_text_features(csv_path, output_path, batch_size=32, num_workers=4):
    print(f"Processing: {csv_path}")

    data = pd.read_csv(csv_path)

    dataset = TextDataset(data, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print("Starting feature extraction...")
    start_time = time.time()

    feature_list = []
    hash_id_list = []
    emotion_list = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        batch_hash_ids = batch.pop('hash_id')
        batch_emotions = batch.pop('emotion')

        batch_features = extract_features_batch(batch)

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
    train_csv_path = '/root/Downloads/train_pre_crowd.csv'
    train_output_path = "/root/maidung/rubert_train_text_features_unique_new.csv"
    train_time = extract_text_features(train_csv_path, train_output_path)

    test_csv_path = '/root/Downloads/test_pre_crowd.csv'
    test_output_path = "/root/maidung/rubert_test_text_features_unique_new.csv"
    test_time = extract_text_features(test_csv_path, test_output_path)

    print("\nSummary:")
    print(f"- Train feature extraction: {train_time:.2f} seconds")
    print(f"- Test feature extraction: {test_time:.2f} seconds")
    print(f"- Total time: {train_time + test_time:.2f} seconds")


if __name__ == "__main__":
    main()

