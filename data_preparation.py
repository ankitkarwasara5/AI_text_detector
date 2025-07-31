import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# --- Configuration ---
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

class AIGeneratedTextDataset(Dataset):
    """
    UPDATED: Handles the simpler structure of train_v2.csv.
    """
    def __init__(self, dataframe, tokenizer, max_token_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        # Use the 'text' and 'label' columns from train_v2.csv
        text_to_encode = sample['text']
        label = sample['label']
        
        encoding = self.tokenizer.encode_plus(
            text_to_encode,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }