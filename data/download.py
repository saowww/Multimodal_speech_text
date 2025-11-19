import requests
import json
import os


def download_dataset(dataset_url, output_path='dataset.zip'):
    kaggle_path = os.path.expanduser('~/.kaggle/kaggle.json')
    with open(kaggle_path, 'r') as f:
        kaggle_config = json.load(f)

    username = kaggle_config['username']
    key = kaggle_config['key']

    headers = {
        'Authorization': f'Basic {username}:{key}'
    }

    response = requests.get(dataset_url, headers=headers)

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Dataset downloaded successfully to {output_path}")
        return True
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return False


def main():
    dataset_url = 'https://www.kaggle.com/api/v1/datasets/download/dngmai1234/crowd-dusha-dataset'
    download_dataset(dataset_url)


if __name__ == "__main__":
    main()

