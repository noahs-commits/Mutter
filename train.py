import dataclasses
import itertools
import math
import os
import time
import torch
import torch.nn as nn
import timed_tokenizer
from tqdm import tqdm
from dataset import time_stamped_audio
from model import hyperparameters
from model.model import transcription_model
from timed_tokenizer import audio_tokenizer
from torch.utils.data import DataLoader, Dataset
from collate_fn import collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training Hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 4       # Adjust based on GPU memory
NUM_EPOCHS = 10
PAD_TOKEN_ID = timed_tokenizer.ignore_id   

     # Replace with your actual pad token ID
GRADIENT_ACCUMULATION_STEPS = 4 # Simulate timed_text_data_setlarger batch size
CLIP_GRAD_NORM = 1.0    # Gradient clipping value
SAVE_CHECKPOINT_DIR = "model_checkpoints"
BEST_MODEL_PATH = os.path.join(SAVE_CHECKPOINT_DIR, "best_model.pt")


# datasets
train_dataset = time_stamped_audio("train")
val_dataset =  time_stamped_audio("test")


hp=hyperparameters.hyperparameters(
    transformer_dim=384,
    layer_count=4, 
    head_count=6,
    dim_feedforward=768,
    activation='gelu',
    dropout=0.1,
    max_audio_len=512,
    max_audio_lookback=10,
    max_token=512,
    n_tokens=timed_tokenizer.token_count
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, hp.head_count,80), 
    pin_memory=device == "cuda",
    num_workers=min(4, os.cpu_count() // 2) if device == "cuda" else 0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle validation data
    collate_fn=lambda b: collate_fn(b, hp.head_count,80),
    pin_memory=device == "cuda",
    num_workers=min(4, os.cpu_count() // 2) if device == "cuda" else 0
)

try:
    os.makedirs(SAVE_CHECKPOINT_DIR, exist_ok=True)
    
    #model,best_val_loss=my_load(BEST_MODEL_PATH)
    model,best_val_loss=transcription_model.load(BEST_MODEL_PATH)
    print(f"Model loaded from {BEST_MODEL_PATH}")
except (FileNotFoundError,PermissionError):
    print(f"Checkpoint not found. Initializing a new model.")
    # Initialize the model
    model=transcription_model(hp)
    best_val_loss=float("inf")

model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, grad_accum_steps, clip_norm):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    optimizer.zero_grad() # Zero gradients at the start of the epoch / first step

    try:
        progress_bar = tqdm(enumerate(dataloader), total=num_batches, desc=f"Epoch {epoch+1} Training")

        for step, batch in progress_bar:
            input_text    = batch['input_text' ].to(device)
            desired_output= batch['output_text'].to(device)
            audio         = batch["audio"      ].to(device)
            tgt_mask      = batch["tgt_mask"   ].to(device)
            memory_mask   = batch["memory_mask"].to(device)
            can_be_end    = batch["can_be_end" ].to(device)

            assert not (torch.eq(desired_output,timed_tokenizer.end_id          )&~can_be_end).any(), "Output text contains end_id where can_be_end is False"
            assert not (torch.eq(desired_output,timed_tokenizer.add_to_buffer_id)& can_be_end).any(), "Output text contains add_to_buffer_id where can_be_end is True"

            
            # Forward pass
            outputs = model(audio,input_text,tgt_mask,memory_mask,can_be_end)
            # outputs shape: (seq_len, batch_size, vocab_size)
            # Calculate loss
            # CrossEntropyLoss expects (N, C) and (N,)
            # N = batch_size * seq_len, C = vocab_size
            outputs = outputs.transpose(0, 1) # Shape: (batch_size, seq_len, vocab_size)

            assert outputs.shape==desired_output.shape+(model.hp.n_tokens,), f"Output shape {outputs.shape} does not match desired output shape {desired_output.shape}"

            output_unrolled         = outputs.flatten(end_dim=1) # Shape: (batch_size * seq_len, vocab_size)
            desired_output_unrolled = desired_output.flatten() # Shape: (batch_size * seq_len)
            can_be_end_unrolled     = can_be_end.flatten() # Shape: (batch_size * seq_len)

            assert not (torch.eq(desired_output_unrolled,timed_tokenizer.end_id          )&~can_be_end_unrolled).any(), "Output text contains end_id where can_be_end is False"
            assert not (torch.eq(desired_output_unrolled,timed_tokenizer.add_to_buffer_id)& can_be_end_unrolled).any(), "Output text contains add_to_buffer_id where can_be_end is True"

            loss = criterion(output_unrolled, desired_output_unrolled)

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            total_loss += loss.item() * grad_accum_steps # Unscale for logging

            # Backward pass
            loss.backward()

            # Optimizer step (perform step after accumulation)
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_batches:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

                # Optimizer step
                optimizer.step()

                #Scheduler step
                #scheduler.step()

                # Zero gradients
                optimizer.zero_grad()

                # Update progress bar
                avg_loss = total_loss / (step + 1) # This might be slightly off due to accum steps, but good estimate
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        avg_train_loss = total_loss / num_batches
        return avg_train_loss
    except KeyboardInterrupt:
        print("Training ended Early")
        try:
            return avg_loss
        except UnboundLocalError:
            return float("inf") # If no steps were completed, return inf
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    start_time= time.time()

    with torch.no_grad(): # Disable gradient calculations
        progress_bar = tqdm(dataloader, total=num_batches, desc="Validating")
        for batch in progress_bar:
            input_text    = batch['input_text' ].to(device)
            desired_output= batch['output_text'].to(device)
            audio         = batch["audio"      ].to(device)
            tgt_mask      = batch["tgt_mask"   ].to(device)
            memory_mask   = batch["memory_mask"].to(device)
            can_be_end    = batch["can_be_end" ].to(device)

            
            # Forward pass
            outputs = model(audio,input_text,tgt_mask,memory_mask,can_be_end)
            # outputs shape: (seq_len, batch_size, vocab_size)
            # Calculate loss
            # CrossEntropyLoss expects (N, C) and (N,)
            # N = batch_size * seq_len, C = vocab_size
            outputs = outputs.transpose(0, 1) # Shape: (batch_size, seq_len, vocab_size)

            assert outputs.shape==desired_output.shape+(model.hp.n_tokens,), f"Output shape {outputs.shape} does not match desired output shape {desired_output.shape}"

            output_unrolled         = outputs.flatten(end_dim=1) # Shape: (batch_size * seq_len, vocab_size)
            desired_output_unrolled = desired_output.flatten() # Shape: (batch_size * seq_len)

            # Calculate loss
            loss = criterion(output_unrolled, desired_output_unrolled)
            total_loss += loss.item()

            progress_bar.set_postfix({'val_loss': f'{total_loss / (progress_bar.n + 1):.4f}'})


    avg_val_loss = total_loss / num_batches
    perplexity = math.exp(avg_val_loss) # Perplexity is exp(average cross-entropy loss)
    return avg_val_loss, perplexity


# --- 8. Training Loop ---
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()

    print("-" * 60)
    avg_train_loss = train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch,
        GRADIENT_ACCUMULATION_STEPS,
        CLIP_GRAD_NORM
    )

    
    avg_val_loss, perplexity = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    epoch_duration = time.time() - epoch_start_time

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_duration:.2f}s")
    print(f"\tAverage Training Loss: {avg_train_loss:.4f}")
    print(f"\tAverage Validation Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f}")

    # Save the best model based on validation loss
    if avg_val_loss < best_val_loss:
        print(f"\tValidation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
        best_val_loss = avg_val_loss

        model.save(BEST_MODEL_PATH,best_val_loss)
        print(f"\tBest model saved to {BEST_MODEL_PATH}")
    else:
        print(f"\tValidation loss did not improve from {best_val_loss:.4f}")
        print(f"\tThis model has a Validation loss of {avg_val_loss:.4f}")

        for name_id in itertools.count():
            name = f"worse_model_{name_id}.pt"
            path = os.path.join(SAVE_CHECKPOINT_DIR, name)
            if not os.path.exists(path):
                print(f"\tSaving model to {path}")
                model.save(path,avg_val_loss)
                break


    # Optional: Save checkpoint at the end of each epoch
    # epoch_checkpoint_path = os.path.join(SAVE_CHECKPOINT_DIR, f"epoch_{epoch+1}_checkpoint.pt")
    # torch.save({ ... }, epoch_checkpoint_path)
    # print(f"\tEpoch checkpoint saved to {epoch_checkpoint_path}")


total_training_time = time.time() - start_time
print("-" * 60)
print(f"Training finished in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
print(f"Best validation loss achieved: {best_val_loss:.4f}")
print(f"Best model saved at: {BEST_MODEL_PATH}")

# --- 9. (Optional) Load the best model for inference ---
# print("\nLoading best model for testing...")
# checkpoint = torch.load(BEST_MODEL_PATH)
# model_config = checkpoint['model_config']
# loaded_model = GPT2Model(**model_config) # Recreate model from saved config
# loaded_model.load_state_dict(checkpoint['model_state_dict'])
# loaded_model.to(device)
# loaded_model.eval()
# print("Model loaded successfully.")

# Add inference example here if needed