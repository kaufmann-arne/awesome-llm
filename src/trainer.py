import torch
import logging
from torch.utils.data import DataLoader
from utils import save_checkpoint, load_checkpoint, get_device, setup_optimizer, setup_scheduler, calculate_accuracy, calculate_loss
from dataloader import create_dataloader

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=32, lr=5e-5, epochs=3, device=None):
        self.model = model
        self.device = device if device else get_device()  # Use GPU if available, else CPU
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        
        # Move model to the correct device (GPU/CPU)
        self.model.to(self.device)

        # Initialize DataLoader
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # Set up optimizer and scheduler
        self.optimizer = setup_optimizer(self.model, self.lr)
        self.total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = setup_scheduler(self.optimizer, self.total_steps)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()  # Set model to training mode
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            logging.info(f"Epoch {epoch + 1}/{self.epochs}")

            # Training loop
            for batch in self.train_dataloader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch, labels=batch)
                loss = calculate_loss(outputs.logits, batch)
                total_loss += loss.item()

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Calculate accuracy (optional)
                accuracy = calculate_accuracy(outputs.logits, batch)
                correct_predictions += accuracy * batch.size(0)
                total_samples += batch.size(0)

            avg_loss = total_loss / len(self.train_dataloader)
            avg_accuracy = correct_predictions / total_samples
            logging.info(f"Epoch {epoch + 1}: Training Loss={avg_loss:.4f}, Training Accuracy={avg_accuracy:.4f}")

            # Save checkpoint at the end of each epoch
            save_checkpoint(self.model, self.optimizer, epoch)

            # Evaluate on validation set after every epoch
            self.evaluate()

    def evaluate(self):
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():  # No gradients needed for evaluation
            for batch in self.val_dataloader:
                batch = batch.to(self.device)

                # Forward pass
                outputs = self.model(batch, labels=batch)
                loss = calculate_loss(outputs.logits, batch)
                total_loss += loss.item()

                # Calculate accuracy
                accuracy = calculate_accuracy(outputs.logits, batch)
                correct_predictions += accuracy * batch.size(0)
                total_samples += batch.size(0)

        avg_loss = total_loss / len(self.val_dataloader)
        avg_accuracy = correct_predictions / total_samples
        logging.info(f"Validation Loss={avg_loss:.4f}, Validation Accuracy={avg_accuracy:.4f}")

    def save_model(self, save_dir="checkpoints"):
        save_checkpoint(self.model, self.optimizer, self.epochs, save_dir)

    def load_model(self, checkpoint_path):
        self.model, self.optimizer, _ = load_checkpoint(self.model, self.optimizer, checkpoint_path, self.device)

if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForCausalLM
    from datasets import load_dataset

    # Load dataset
    dataset = load_dataset("wikitext", split="train")
    val_dataset = load_dataset("wikitext", split="validation")

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Initialize trainer
    trainer = Trainer(model, dataset, val_dataset, batch_size=8, epochs=5)

    # Train the model
    trainer.train()
 
