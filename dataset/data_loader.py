import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for _, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data['text'])
        print(f'finish loading data from {path}, total {len(samples)} samples')
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        text = f"{self.tokenizer.bos_token}{sample}{self.tokenizer.eos_token}"
        encoding = self.tokenizer(text,
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='np')
        input_ids = np.squeeze(encoding.input_ids)
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = np.concatenate(
            [input_ids[:-1],
             np.array([self.tokenizer.pad_token_id])], axis=-1)
        Y = np.concatenate(
            [input_ids[1:],
             np.array([self.tokenizer.pad_token_id])], axis=-1)

        X = torch.tensor(X, dtype=torch.long)
        Y = torch.tensor(Y, dtype=torch.long)
        return X, Y, loss_mask
