import torch
from torch.utils.data import Dataset, DataLoader
from transformers import HuggingFaceAutoTokenizer  
from datasets import load_dataset

class AwesomeLLMDataset(Dataset):
    """
    Custom Dataset for loading and processing the text data for AwesomeLLM.
    """
    def __init__(self, dataset_name: str, tokenizer: AutoTokenizer, block_size: int):
        """
        Args:
            dataset_name (str): Name of the dataset to load (e.g., 'wikitext' or custom dataset).
            tokenizer (AutoTokenizer): The tokenizer used for tokenizing the text.
            block_size (int): The maximum sequence length for the AwesomeLLM model.
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Load dataset using HuggingFace's datasets library
        self.dataset = load_dataset(dataset_name, split='train')  # Adjust to dataset
        self.text_data = self.dataset['text']  # Assuming  dataset has a 'text' column

    def __len__(self):
        # Return the number of examples in the dataset
        return len(self.text_data)

    def __getitem__(self, idx):
        # Get the text for the given index
        text = self.text_data[idx]
        
        # Tokenize the text, truncating or padding to fit within the block size
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.block_size, return_tensors='pt')
        
        # Return input_ids as a tensor
        return encoding['input_ids'].squeeze()  # Remove the extra batch dimension

def create_dataloader(dataset_name: str, batch_size: int, block_size: int = 128):
    """
    Create DataLoader for loading batches of data for training.
    
    Args:
        dataset_name (str): Dataset name to load.
        batch_size (int): Batch size for the DataLoader.
        block_size (int): Maximum sequence length.
    
    Returns:
        DataLoader: A DataLoader instance.
    """
    # Initialize Hugging Face's AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  #replace this with any model 
    
    # Create custom dataset
    dataset = AwesomeLLMDataset(dataset_name, tokenizer, block_size)
    
    # Create DataLoader for batching data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader

if __name__ == "__main__":
    # Example usage of the dataloader
    dataset_name = "wikitext"  # Choose your dataset
    batch_size = 32            # Example batch size
    dataloader = create_dataloader(dataset_name, batch_size)

    # Example: iterate over DataLoader
    for batch in dataloader:
        print(batch.shape)  # Should be [batch_size, block_size]
        break
