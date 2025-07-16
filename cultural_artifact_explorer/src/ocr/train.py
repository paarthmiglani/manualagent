# src/ocr/train.py
import yaml
import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.ocr.model_definition import CRNN
from src.ocr.dataset import OCRDataset, ocr_collate_fn
from src.ocr.utils import load_char_list
from src.ocr.postprocess import ctc_decode_predictions
from src.utils.metrics import calculate_cer, calculate_wer

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    sys.exit(1)

class OCRTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_config = self.config.get('model', {})
        self.train_config = self.config.get('training', {})
        self.output_dir = self.train_config.get('output_dir', 'models/ocr/')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"OCRTrainer initialized. Model outputs will be saved to: {self.output_dir}")
        self._setup_device()
        self._load_char_list()
        self._build_model()
        self._load_data()
        self._setup_training_components()

    def _setup_device(self):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)!")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU!")
        else:
            self.device = torch.device("cpu")
            print("Using CPU.")
        print(f"Using device: {self.device}")

    def _load_char_list(self):
        char_list_path = self.model_config.get('char_list_path')
        if not char_list_path:
            raise ValueError("Path to 'char_list_path' must be specified in model config.")
        self.char_list = load_char_list(char_list_path)
        self.ix_to_char = {i: char for i, char in enumerate(self.char_list)}
        self.ix_to_char[0] = '<BLANK>'

    def _build_model(self):
        print("Building OCR model...")
        num_classes = len(self.char_list)
        self.model = CRNN(
            img_channels=self.model_config.get('input_channels', 1),
            num_classes=num_classes,
            rnn_hidden_size=self.model_config.get('rnn_hidden_size', 256),
            rnn_num_layers=self.model_config.get('num_rnn_layers', 2),
            dropout=self.model_config.get('rnn_dropout', 0.5)
        ).to(self.device)
        print("Model built successfully.")

    def _load_data(self):
        print("Loading OCR training and validation data...")
        common_dataset_params = {
            'char_list_path': self.model_config['char_list_path'],
            'image_height': self.model_config.get('input_height', 32),
            'image_width': self.model_config.get('input_width', 256),
            'binarize': self.config.get('preprocessing', {}).get('binarize', False)
        }
        train_dataset = OCRDataset(
            annotations_file=self.train_config['annotations_file'],
            img_dir=self.train_config['dataset_path'],
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
        self.val_loader = None
        val_ann_file = self.train_config.get('validation_annotations_file')
        val_img_dir = self.train_config.get('validation_dataset_path')
        if val_ann_file and val_img_dir:
            val_dataset = OCRDataset(
                annotations_file=val_ann_file,
                img_dir=val_img_dir,
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
        print("Setting up optimizer, scheduler, and loss function...")
        self.criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        optimizer_name = self.train_config.get('optimizer', 'Adam').lower()
        lr = self.train_config.get('learning_rate', 0.001)
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.train_config.get('weight_decay', 0.0))
        else:
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)

    def train_epoch(self, epoch_num):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_num+1} Training", ncols=100)
        for batch_i, (images, labels, _, label_lengths) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device).long()
            label_lengths = label_lengths.to(self.device).long()
            self.optimizer.zero_grad()
            log_probs = self.model(images)
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=log_probs.size(0),
                dtype=torch.long,
                device=self.device
            )
            loss = self.criterion(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.get('grad_clip_norm', 5))
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Batch": f"{batch_i+1}/{len(self.train_loader)}"
            })
        avg_epoch_loss = total_loss / len(self.train_loader)
        print(f"\nEnd of Epoch {epoch_num+1}, Average Training Loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def validate_epoch(self, epoch_num):
        if not self.val_loader:
            print("No validation loader configured. Skipping validation.")
            return
        print(f"Running validation for Epoch {epoch_num+1}...")
        self.model.eval()
        total_cer, total_wer, val_loss = 0, 0, 0
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch_num+1} Validation", ncols=100)
            for images, labels, _, label_lengths in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                label_lengths = label_lengths.to(self.device).long()
                log_probs = self.model(images)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=log_probs.size(0),
                    dtype=torch.long,
                    device=self.device
                )
                loss = self.criterion(log_probs, labels, input_lengths, label_lengths)
                val_loss += loss.item()
                preds = log_probs.permute(1, 0, 2).cpu().numpy() # (B, T, C)
                batch_size = len(preds)
                for i in range(batch_size):
                    pred_text, _ = ctc_decode_predictions(preds[i], self.char_list, blank_idx=0)
                    true_label_indices = labels[label_lengths[:i].sum():label_lengths[:i+1].sum()].tolist()
                    true_text = "".join([self.ix_to_char.get(ix, '') for ix in true_label_indices])
                    total_cer += calculate_cer(true_text, pred_text)
                    total_wer += calculate_wer(true_text, pred_text)
        avg_val_loss = val_loss / len(self.val_loader)
        avg_cer = total_cer / len(self.val_loader.dataset)
        avg_wer = total_wer / len(self.val_loader.dataset)
        print(f"Validation Results Epoch {epoch_num+1}: Loss={avg_val_loss:.4f}, CER={avg_cer:.4f}, WER={avg_wer:.4f}")
        return avg_val_loss

    def run_training(self):
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
    if not os.path.exists(args.config):
        print("Config file not found. This script requires a valid configuration.")
        sys.exit(1)
    trainer = OCRTrainer(config_path=args.config)
    trainer.run_training()

if __name__ == '__main__':
    main()
