import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import pandas as pd
from torch.optim import AdamW

# Import our custom modules
from data_preparation import AIGeneratedTextDataset
from model import BERTClassifier, PRE_TRAINED_MODEL_NAME

# --- Configuration ---
EPOCHS = 3 # Can start with 3 epochs for the larger dataset
BATCH_SIZE = 8
MAX_TOKEN_LEN = 512
LEARNING_RATE = 2e-5
MODEL_SAVE_PATH = 'best_model_state_v2.bin' # Changed model name

# UPDATE THIS PATH to your new dataset file
DATASET_PATH = 'train_v2.csv'

# --- Helper functions train_epoch() and eval_model() remain the same ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()
    losses = []
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return sum(losses) / len(losses)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = sum(losses) / len(losses)
    print("\nValidation Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Human', 'AI']))
    return avg_loss, accuracy

# --- Main Training Script ---
if __name__ == '__main__':

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")


    # --- SIMPLIFIED: Load the dataset directly ---
    # The undersampling logic is no longer needed.
    dataframe = pd.read_csv(DATASET_PATH)
    print("Dataset loaded. Info:")
    print(dataframe['label'].value_counts())
    
    # --- The rest of the script is largely the same ---
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    full_dataset = AIGeneratedTextDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_token_len=MAX_TOKEN_LEN
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BERTClassifier(n_classes=2).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'\n--- Epoch {epoch + 1}/{EPOCHS} ---')
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device, scheduler)
        print(f'Training loss: {train_loss:.4f}')
        val_loss, val_accuracy = eval_model(model, val_dataloader, loss_fn, device)
        print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_accuracy = val_accuracy
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")

    print("\n--- Training complete ---")
    print(f"Best validation accuracy: {best_accuracy:.4f}")