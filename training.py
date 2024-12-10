import os
import json
import torch
import logging
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import wandb  # For experiment tracking
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
from datasets import load_dataset
import requests
from io import BytesIO

from text_image_token_processor_1 import PaliGemmaProcessor
from decoder_1 import PaliGemmaForConditionalGeneration
from utils import load_hf_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalDataset(Dataset):
    """
    Dataset class for handling image-text pairs using HuggingFace datasets.
    """
    def __init__(
        self, 
        processor: PaliGemmaProcessor,
        split: str = "train",
        max_length: int = 512,
        max_samples: int = None  # Limit samples for quick testing
    ):
        # Load Flickr8k dataset
        self.dataset = load_dataset("nlphuji/flickr8k", split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(max_samples))
        
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(BytesIO(requests.get(item['image_url']).content)).convert('RGB')
        text = item['caption']
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': inputs['input_ids'].squeeze(0)
        }

class Trainer:
    def __init__(
        self,
        model: PaliGemmaForConditionalGeneration,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        device: str,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        log_to_wandb: bool = True
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.log_to_wandb = log_to_wandb
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if use_amp else None
    
    def train_epoch(self, epoch: int):
        """Run one epoch of training"""
        self.model.train()
        total_loss = 0
        
        # Create progress bar
        pbar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]  # Model will automatically shift labels
                )
                loss = outputs.loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            pbar.set_postfix({'loss': avg_loss})
            
            # Log to wandb
            if self.log_to_wandb and step % 100 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'train_avg_loss': avg_loss,
                    'epoch': epoch,
                    'step': step
                })
        
        return total_loss / len(self.train_dataloader)
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model on validation set"""
        if not self.val_dataloader:
            return None
            
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        return avg_loss

def main():
    # Initialize wandb
    wandb.init(project="paligemma-training")
    
    # Training configuration
    config = {
        'model_path': "paligemma-3b-pt-224/",
        'train_json': "data/train.json",
        'val_json': "data/val.json",
        'image_dir': "data/images/",
        'output_dir': "trained_model/",
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'num_epochs': 3,
        'use_amp': True,
        'max_grad_norm': 1.0,
    }
    
    # Load model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_hf_model(config['model_path'], device)
    processor = PaliGemmaProcessor(
        tokenizer,
        num_image_tokens=model.config.vision_config.num_image_tokens,
        image_size=model.config.vision_config.image_size
    )
    
    # Create datasets and dataloaders
    train_dataset = MultiModalDataset(
        processor=processor,
        split="train",
        max_samples=100  # Start with small number for testing
    )
    
    val_dataset = MultiModalDataset(
        processor=processor,
        split="validation",
        max_samples=50  # Start with small number for testing
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) if val_dataset else None
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        use_amp=config['use_amp'],
        max_grad_norm=config['max_grad_norm']
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        # Train
        train_loss = trainer.train_epoch(epoch)
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if val_dataloader:
            val_loss = trainer.evaluate()
            logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(os.path.join(config['output_dir'], 'best_model'))
                tokenizer.save_pretrained(os.path.join(config['output_dir'], 'best_model'))
        
        # Save checkpoint
        checkpoint_dir = os.path.join(config['output_dir'], f'checkpoint-{epoch}')
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss if val_dataloader else None
        })
    
    wandb.finish()

if __name__ == "__main__":
    main() 