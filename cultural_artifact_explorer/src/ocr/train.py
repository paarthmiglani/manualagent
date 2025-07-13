# src/ocr/train.py
# Script for OCR model training, now with validation.

import yaml
import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import model, dataset, and utilities from our source files
from .model_definition import CRNN
from .dataset import OCRDataset, ocr_collate_fn
from .utils import load_char_list
from .postprocess import ctc_decode_predictions
from ..utils.metrics import calculate_cer, calculate_wer # Import from top-level utils

class OCRTrainer:
    def __init__(self, config_path):
        """Initializes the trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config.get('model', {})
        self.train_config = self.config.get('training', {})

        self.output_dir = self.train_config.get('output_dir', 'models/ocr/default_run/')
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"OCRTrainer initialized. Model outputs will be saved to: {self.output_dir}")
        self._setup_device()
        self._load_char_list() # Load char list before building model and data
        self._build_model()
        self._load_data()
        self._setup_training_components()

    def _setup_device(self):
        """Sets up the device for training (CPU or GPU)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _load_char_list(self):
        """Loads the character list to be used by the model and dataset."""
        char_list_path = self.model_config.get('char_list_path')
        if not char_list_path:
            raise ValueError("Path to 'char_list_path' must be specified in model config.")
        self.char_list = load_char_list(char_list_path)
        self.ix_to_char = {i + 1: char for i, char in enumerate(self.char_list)}
        self.ix_to_char[0] = '<BLANK>'


    def _build_model(self):
        """Builds the CRNN model based on configuration."""
        print("Building OCR model...")
        num_classes = len(self.char_list) + 1  # +1 for the blank token required by CTC

        self.model = CRNN(
            img_channels=self.model_config.get('input_channels', 1),
            num_classes=num_classes,
            rnn_hidden_size=self.model_config.get('rnn_hidden_size', 256),
            rnn_num_layers=self.model_config.get('num_rnn_layers', 2),
            dropout=self.model_config.get('rnn_dropout', 0.5)
        ).to(self.device)
        print("Model built successfully.")

    def _load_data(self):
        """Loads training and validation datasets."""
        print("Loading OCR training and validation data...")
        common_dataset_params = {
            'char_list_path': self.model_config['char_list_path'],
            'image_height': self.model_config.get('input_height', 32),
            'image_width': self.model_config.get('input_width', 256),
            'binarize': self.config.get('preprocessing', {}).get('binarize', False)
        }

        # Training dataset
        train_dataset = OCRDataset(
            annotations_file=self.train_config['annotations_file'],
            **common_dataset_params
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            num_workers=self.train_config.get('num_workers', 2),
            collate_fn=ocr_collate_fn,
            pin_memory=True
        )
        print(f"  Training data loaded: {len(train_dataset)} samples.")

        # Validation dataset
        self.val_loader = None
        val_ann_file = self.train_config.get('validation_annotations_file')
        if val_ann_file:
            val_dataset = OCRDataset(
                annotations_file=val_ann_file,
                **common_dataset_params
            )
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.train_config.get('validation_batch_size', self.train_config['batch_size']),
                shuffle=False,
                num_workers=self.train_config.get('num_workers', 2),
                collate_fn=ocr_collate_fn,
                pin_memory=True
            )
            print(f"  Validation data loaded: {len(val_dataset)} samples.")
        else:
            print("  No validation dataset configured.")


    def _setup_training_components(self):
        """Sets up the loss function, optimizer, and scheduler."""
        print("Setting up optimizer, scheduler, and loss function...")
        self.criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True).to(self.device)
        optimizer_name = self.train_config.get('optimizer', 'Adam').lower()
        lr = self.train_config.get('learning_rate', 0.001)

        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.train_config.get('weight_decay', 0.0))
        else:
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)

    def train_epoch(self, epoch_num):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0
        for i, (images, labels, _, label_lengths) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch_num+1} Training")):
            images, labels, label_lengths = images.to(self.device), labels.to(self.device), label_lengths.to(self.device)
            self.optimizer.zero_grad()
            log_probs = self.model(images)
            input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long)
            loss = self.criterion(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.get('grad_clip_norm', 5))
            self.optimizer.step()
            total_loss += loss.item()

        avg_epoch_loss = total_loss / len(self.train_loader)
        print(f"End of Epoch {epoch_num+1}, Average Training Loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def validate_epoch(self, epoch_num):
        """Runs a single validation epoch and computes metrics."""
        if not self.val_loader:
            print("No validation loader configured. Skipping validation.")
            return

        print(f"Running validation for Epoch {epoch_num+1}...")
        self.model.eval()
        total_cer, total_wer, val_loss = 0, 0, 0

        with torch.no_grad():
            for images, labels, _, label_lengths in tqdm(self.val_loader, desc=f"Epoch {epoch_num+1} Validation"):
                images, labels, label_lengths = images.to(self.device), labels.to(self.device), label_lengths.to(self.device)
                log_probs = self.model(images)
                input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long)
                loss = self.criterion(log_probs, labels, input_lengths, label_lengths)
                val_loss += loss.item()

                # Decode predictions for metrics calculation
                preds = log_probs.permute(1, 0, 2).cpu().numpy() # (B, T, C)

                for i in range(len(preds)):
                    pred_text, _ = ctc_decode_predictions(preds[i], self.char_list, blank_idx=0)

                    # Decode ground truth label
                    true_label_indices = labels[i][:label_lengths[i]].tolist()
                    true_text = "".join([self.ix_to_char[ix] for ix in true_label_indices])

                    total_cer += calculate_cer(true_text, pred_text)
                    total_wer += calculate_wer(true_text, pred_text)

        avg_val_loss = val_loss / len(self.val_loader)
        avg_cer = total_cer / len(self.val_loader.dataset)
        avg_wer = total_wer / len(self.val_loader.dataset)

        print(f"Validation Results Epoch {epoch_num+1}: Loss={avg_val_loss:.4f}, CER={avg_cer:.4f}, WER={avg_wer:.4f}")
        return avg_val_loss

    def run_training(self):
        """Main training loop."""
        print("\n--- Starting OCR Model Training ---")
        num_epochs = self.train_config.get('epochs', 1)

        for epoch in range(num_epochs):
            self.train_epoch(epoch)
            val_metric = self.validate_epoch(epoch)

            if self.scheduler and val_metric is not None:
                self.scheduler.step(val_metric)

            if (epoch + 1) % self.train_config.get('save_epoch_interval', 1) == 0:
                save_path = os.path.join(self.output_dir, f"crnn_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Model checkpoint saved to: {save_path}")

        print("\n--- OCR Model Training Finished ---")

def main():
    parser = argparse.ArgumentParser(description="Train an OCR model with validation.")
    parser.add_argument('--config', type=str, default="configs/ocr.yaml", help="Path to the OCR configuration YAML file.")
    args = parser.parse_args()

    # Simple check for dummy run
    if not os.path.exists(args.config):
        print("Config file not found. This script requires a valid configuration.")
        sys.exit(1)

    try:
        # Add tqdm for progress bars
        from tqdm import tqdm as tqdm_check
    except ImportError:
        print("Please install tqdm: pip install tqdm")
        sys.exit(1)

    trainer = OCRTrainer(config_path=args.config)
    trainer.run_training()

if __name__ == '__main__':
    # Add tqdm for progress bars if not already installed
    try:
        from tqdm import tqdm
    except ImportError:
        print("Tqdm not found. Please install it: pip install tqdm")
        sys.exit(1)

    main()
