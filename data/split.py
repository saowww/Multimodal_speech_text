import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def split_data(audio_train_path, audio_test_path, text_train_path, text_test_path, output_dir, test_size=0.2, random_state=42):
    print("Reading feature sets...")
    audio_train_df = pd.read_csv(audio_train_path)
    audio_test_df = pd.read_csv(audio_test_path)
    text_train_df = pd.read_csv(text_train_path)
    text_test_df = pd.read_csv(text_test_path)

    print("Merging feature sets...")
    audio_df = pd.concat([audio_train_df, audio_test_df])
    text_df = pd.concat([text_train_df, text_test_df])

    print("Ensuring data order...")
    audio_df = audio_df.sort_values(by='name')
    text_df = text_df.sort_values(by='name')

    print("Checking hash_id matching...")
    audio_ids = set(audio_df['name'].values)
    text_ids = set(text_df['name'].values)
    common_ids = audio_ids.intersection(text_ids)

    print(f"Total hash_ids in audio set: {len(audio_ids)}")
    print(f"Total hash_ids in text set: {len(text_ids)}")
    print(f"Hash_ids in both sets: {len(common_ids)}")

    audio_df = audio_df[audio_df['name'].isin(common_ids)]
    text_df = text_df[text_df['name'].isin(common_ids)]

    audio_df = audio_df.set_index('name').loc[list(common_ids)].reset_index()
    text_df = text_df.set_index('name').loc[list(common_ids)].reset_index()

    print("Checking order after sorting...")
    is_order_correct = (audio_df['name'] == text_df['name']).all()
    print(f"Hash_id order matches: {is_order_correct}")

    if not is_order_correct:
        print("Error: Hash_id order does not match!")
        print("Re-sorting using alternative method...")
        common_ids_list = sorted(list(common_ids))
        audio_df = audio_df.loc[audio_df['name'].isin(common_ids_list)].sort_values(
            by='name').reset_index(drop=True)
        text_df = text_df.loc[text_df['name'].isin(common_ids_list)].sort_values(
            by='name').reset_index(drop=True)

        is_order_correct = (audio_df['name'] == text_df['name']).all()
        print(f"Order after re-sorting: {is_order_correct}")

        if not is_order_correct:
            raise ValueError("Cannot ensure hash_id order!")

    emotions = audio_df['emotion'].values

    audio_features = audio_df.drop(['name', 'emotion'], axis=1)
    text_features = text_df.drop(['name', 'emotion'], axis=1)

    print(f"Splitting dataset with {test_size*100}% test ratio by emotion group...")

    indices_df = pd.DataFrame({
        'index': np.arange(len(audio_df)),
        'emotion': emotions
    })

    train_indices = []
    test_indices = []

    for emotion in np.unique(emotions):
        emotion_indices = indices_df[indices_df['emotion']
                                     == emotion]['index'].values

        if len(emotion_indices) == 1:
            train_indices.extend(emotion_indices)
            print(
                f"Warning: Emotion '{emotion}' has only 1 sample, added to training set")
            continue

        emotion_train, emotion_test = train_test_split(
            emotion_indices,
            test_size=test_size,
            random_state=random_state
        )

        train_indices.extend(emotion_train)
        test_indices.extend(emotion_test)

        print(
            f"Emotion '{emotion}': {len(emotion_train)} train samples, {len(emotion_test)} test samples")

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    audio_train = audio_features.iloc[train_indices]
    audio_test = audio_features.iloc[test_indices]
    text_train = text_features.iloc[train_indices]
    text_test = text_features.iloc[test_indices]

    train_names = audio_df['name'].iloc[train_indices]
    test_names = audio_df['name'].iloc[test_indices]
    train_emotions = audio_df['emotion'].iloc[train_indices]
    test_emotions = audio_df['emotion'].iloc[test_indices]

    audio_train['name'] = train_names.values
    audio_train['emotion'] = train_emotions.values
    audio_test['name'] = test_names.values
    audio_test['emotion'] = test_emotions.values

    text_train['name'] = train_names.values
    text_train['emotion'] = train_emotions.values
    text_test['name'] = test_names.values
    text_test['emotion'] = test_emotions.values

    print("Saving processed sets...")
    os.makedirs(output_dir, exist_ok=True)

    audio_train.to_csv(os.path.join(
        output_dir, "audio_train_features.csv"), index=False)
    audio_test.to_csv(os.path.join(
        output_dir, "audio_test_features.csv"), index=False)
    text_train.to_csv(os.path.join(
        output_dir, "text_train_features.csv"), index=False)
    text_test.to_csv(os.path.join(
        output_dir, "text_test_features.csv"), index=False)

    train_names.to_csv(os.path.join(
        output_dir, "train_hash_ids.csv"), index=False)
    test_names.to_csv(os.path.join(
        output_dir, "test_hash_ids.csv"), index=False)

    train_emotions.to_csv(os.path.join(
        output_dir, "train_emotions.csv"), index=False)
    test_emotions.to_csv(os.path.join(
        output_dir, "test_emotions.csv"), index=False)

    print("\nStatistics:")
    print(f"Total samples: {len(audio_df)}")
    print(
        f"Training samples: {len(audio_train)} ({len(audio_train)/len(audio_df)*100:.1f}%)")
    print(
        f"Test samples: {len(audio_test)} ({len(audio_test)/len(audio_df)*100:.1f}%)")

    print("\nDetailed emotion distribution:")

    emotion_stats = []
    for emotion in np.unique(emotions):
        total_count = sum(audio_df['emotion'] == emotion)
        train_count = sum(train_emotions == emotion)
        test_count = sum(test_emotions == emotion)

        emotion_stats.append({
            'Emotion': emotion,
            'Total samples': total_count,
            'Train samples': train_count,
            'Test samples': test_count,
            'Train ratio': f"{train_count/total_count*100:.1f}%",
            'Test ratio': f"{test_count/total_count*100:.1f}%"
        })

    stats_df = pd.DataFrame(emotion_stats)
    print(stats_df.to_string(index=False))

    print("\nOverall emotion distribution:")
    train_dist = pd.Series(train_emotions).value_counts(normalize=True) * 100
    test_dist = pd.Series(test_emotions).value_counts(normalize=True) * 100

    print("Training set:")
    for emo, pct in train_dist.items():
        print(f"  {emo}: {pct:.1f}%")

    print("Test set:")
    for emo, pct in test_dist.items():
        print(f"  {emo}: {pct:.1f}%")

    print("\nProcessing completed!")

    return {
        'audio_train': audio_train,
        'audio_test': audio_test,
        'text_train': text_train,
        'text_test': text_test
    }


def main():
    audio_train_path = "/root/maidung/wav2vec2_train_audio_features_unique_new.csv"
    audio_test_path = "/root/maidung/wav2vec2_test_audio_features_unique_new.csv"
    text_train_path = "/root/maidung/rubert_train_text_features_unique_new.csv"
    text_test_path = "/root/maidung/rubert_test_text_features_unique_new.csv"

    output_dir = "/root/maidung/merged_features/"

    split_data(audio_train_path, audio_test_path, text_train_path, text_test_path, output_dir)


if __name__ == "__main__":
    main()

