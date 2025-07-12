# src/ocr/train.py
# Script for OCR model training.

import yaml
import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import model, dataset, and utilities from our source files
from .model_definition import CRNN
from .dataset import OCRDataset, ocr_collate_fn
from .utils import load_char_list # For getting vocab size

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
        self._build_model()
        self._load_data()
        self._setup_training_components()

    def _setup_device(self):
        """Sets up the device for training (CPU or GPU)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _build_model(self):
        """Builds the CRNN model based on configuration."""
        print("Building OCR model...")
        char_list_path = self.model_config['char_list_path']
        char_list = load_char_list(char_list_path)
        num_classes = len(char_list) + 1  # +1 for the blank token required by CTC

        self.model = CRNN(
            img_channels=self.model_config.get('input_channels', 1),
            num_classes=num_classes,
            rnn_hidden_size=self.model_config.get('rnn_hidden_size', 256),
            rnn_num_layers=self.model_config.get('num_rnn_layers', 2),
            dropout=self.model_config.get('rnn_dropout', 0.5) # Example new config key
        ).to(self.device)
        print("Model built successfully.")
        print(f"  Number of classes: {num_classes}")

    def _load_data(self):
        """Loads training and validation datasets."""
        print("Loading OCR training and validation data...")
        # Assuming annotations are in a single file and we might split it or use separate files.
        # For simplicity, let's assume the config points to a training annotation file.
        # Validation would ideally use a separate file.
        train_dataset = OCRDataset(
            annotations_file=self.train_config['annotations_file'],
            img_dir=self.train_config['dataset_path'],
            char_list_path=self.model_config['char_list_path'],
            image_height=self.model_config.get('input_height', 32),
            image_width=self.model_config.get('input_width', 128), # Add width to config
            binarize=self.config.get('preprocessing', {}).get('binarize', False)
        )

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            num_workers=self.train_config.get('num_workers', 2), # Add num_workers to config
            collate_fn=ocr_collate_fn,
            pin_memory=True
        )

        # Placeholder for validation loader
        # val_annotations_file = self.train_config.get('validation_annotations_file')
        # if val_annotations_file:
        #     val_dataset = OCRDataset(...)
        #     self.val_loader = DataLoader(...)
        # else:
        self.val_loader = None
        print("Data loading complete.")

    def _setup_training_components(self):
        """Sets up the loss function, optimizer, and scheduler."""
        print("Setting up optimizer, scheduler, and loss function...")
        self.criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True).to(self.device)

        optimizer_name = self.train_config.get('optimizer', 'Adam').lower()
        lr = self.train_config.get('learning_rate', 0.001)

        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.train_config.get('weight_decay', 0.0))
        elif optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Optional: Learning rate scheduler
        # scheduler_name = self.train_config.get('scheduler', None)
        # if scheduler_name == 'reducelronplateau':
        #     self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        # else:
        self.scheduler = None

        print(f"  Optimizer: {optimizer_name.capitalize()}, Learning Rate: {lr}")
        print(f"  Loss Function: CTCLoss")

    def train_epoch(self, epoch_num):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0

        for i, (images, labels, image_widths, label_lengths) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            image_widths = image_widths.to(self.device)
            label_lengths = label_lengths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            log_probs = self.model(images) # Output shape: (SeqLen, Batch, NumClasses)

            # CTC Loss requires input_lengths to be based on the model's output sequence length.
            # Our model definition implies a downsampling factor of 4.
            # A more robust way is to get this from the model output shape directly.
            input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long).to(self.device)

            # Calculate loss
            loss = self.criterion(log_probs, labels, input_lengths, label_lengths)

            # Backward pass and optimize
            loss.backward()
            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.get('grad_clip_norm', 5))
            self.optimizer.step()

            total_loss += loss.item()

            if (i + 1) % self.train_config.get('log_interval', 10) == 0:
                print(f"  Epoch {epoch_num+1}, Batch [{i+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")

        avg_epoch_loss = total_loss / len(self.train_loader)
        print(f"End of Epoch {epoch_num+1}, Average Training Loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def validate_epoch(self, epoch_num):
        """Runs a single validation epoch."""
        if not self.val_loader:
            print("No validation loader configured. Skipping validation.")
            return

        self.model.eval()
        # Validation logic would go here (e.g., calculate CER/WER or validation loss)
        # ...
        print(f"End of Epoch {epoch_num+1}, Validation placeholder.")


    def run_training(self):
        """Main training loop."""
        print("\n--- Starting OCR Model Training ---")
        num_epochs = self.train_config.get('epochs', 1)

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            train_loss = self.train_epoch(epoch)

            # Run validation if configured
            self.validate_epoch(epoch)

            # Step the scheduler if it's based on validation metric
            if self.scheduler:
                # self.scheduler.step(validation_metric)
                pass

            # Save model checkpoint
            if (epoch + 1) % self.train_config.get('save_epoch_interval', 1) == 0:
                save_path = os.path.join(self.output_dir, f"crnn_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Model checkpoint saved to: {save_path}")

        print("\n--- OCR Model Training Finished ---")

def main():
    parser = argparse.ArgumentParser(description="Train an OCR model.")
    parser.add_argument('--config', type=str, default="configs/ocr.yaml",
                        help="Path to the OCR configuration YAML file.")
    args = parser.parse_args()

    # Create dummy files for placeholder run if they don't exist
    if not os.path.exists(args.config):
        print(f"Warning: Config file not found at {args.config}. Creating a dummy one.")
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        dummy_cfg = {
            'model': {'char_list_path': 'temp_chars.txt', 'input_height': 32, 'input_width': 128},
            'training': {'annotations_file': 'temp_ann.csv', 'dataset_path': 'temp_imgs', 'batch_size': 2, 'epochs': 1, 'num_workers': 0}
        }
        with open(args.config, 'w') as f: yaml.dump(dummy_cfg, f)

        # Create other dummy files based on config
        with open(dummy_cfg['model']['char_list_path'], 'w') as f: f.write('a\nb\nc\n')
        os.makedirs(dummy_cfg['training']['dataset_path'], exist_ok=True)
        pd.DataFrame([{'filename':'dummy1.png', 'text':'a'}]).to_csv(dummy_cfg['training']['annotations_file'], index=False)
        cv2.imwrite(os.path.join(dummy_cfg['training']['dataset_path'], 'dummy1.png'), np.zeros((32,128,3), dtype=np.uint8))


    try:
        trainer = OCRTrainer(config_path=args.config)
        trainer.run_training()
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up dummy files created for placeholder run
        if 'temp_chars.txt' in open(args.config).read():
            os.remove('temp_chars.txt')
            os.remove('temp_ann.csv')
            os.remove('temp_imgs/dummy1.png')
            os.rmdir('temp_imgs')
            # os.remove(args.config) # Optional: remove dummy config too

if __name__ == '__main__':
    # This script now has a more complete training loop.
    # To run, you need a proper config file and dataset.
    # The placeholder `main` function creates dummy files to allow a dry run.
    print("Executing src.ocr.train (with implemented training loop)...")
    main()
