import torch
import os
from torch import optim
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW, get_linear_schedule_with_warmup
import logging

# Function to save a model checkpoint
def save_checkpoint(model, optimizer, epoch, save_dir="checkpoints", filename="model_checkpoint.pth"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")

# Function to load a model checkpoint
def load_checkpoint(model, optimizer, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}")
    return model, optimizer, epoch

# Function to handle device placement (GPU/CPU)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to setup optimizer
def setup_optimizer(model, learning_rate=5e-5):
    # AdamW optimizer typically used for transformers
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    return optimizer

# Function to setup learning rate scheduler
def setup_scheduler(optimizer, total_steps, warmup_steps=0):
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return scheduler

# Logging setup (to track progress and print results)
def setup_logging(log_dir="logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, "training.log")),
                  logging.StreamHandler()]
    )
    logging.info("Logging initialized")

# Function to calculate accuracy (or any other metric)
def calculate_accuracy(predictions, labels):
    preds = torch.argmax(predictions, dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

# Function to calculate loss (cross-entropy)
def calculate_loss(predictions, labels, loss_fn=torch.nn.CrossEntropyLoss()):
    return loss_fn(predictions, labels)

if __name__ == "__main__":
    # Example usage
    pass
 
