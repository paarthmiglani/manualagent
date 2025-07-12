# src/nlp/train_ner.py
# Script for training the Named Entity Recognition (NER) model.

import yaml
import argparse
import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import model, dataset, and utilities from our source files
from .model_definition_ner import BiLSTM_CRF
from .dataset_ner import NERDataset, ner_collate_fn

class NERTrainer:
    def __init__(self, config_path):
        """Initializes the trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.ner_config = self.config.get('ner', {})
        self.train_config = self.ner_config.get('training', {})
        if not self.train_config:
            raise ValueError("Training configuration ('training') not found in 'ner' section of NLP config.")

        self.output_dir = self.train_config.get('output_dir', 'models/nlp/ner/default_run/')
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"NERTrainer initialized. Model outputs will be saved to: {self.output_dir}")
        self._setup_device()
        self._build_vocab_and_tags()
        self._load_data()
        self._build_model()
        self._setup_optimizer()

    def _setup_device(self):
        """Sets up the device for training (CPU or GPU)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _build_vocab_and_tags(self):
        """Builds vocabulary and tag mappings from the training data."""
        print("Building vocabulary and tag set from training data...")
        # In a real scenario, you'd iterate through your training file to build these.
        # For this placeholder, we'll assume they are pre-built or create simple ones.
        # This process ensures that word_to_ix and tag_to_ix are consistent with the data.

        # Placeholder Mappings (in a real implementation, build from `self.train_config['dataset_path']`)
        self.word_to_ix = {'<PAD>': 0, '<UNK>': 1, 'The': 2, 'Taj': 3, 'Mahal': 4, 'is': 5, 'in': 6, 'Agra': 7}
        # For the BiLSTM-CRF, special START and STOP tags are required.
        self.tag_to_ix = {"O": 0, "B-MONUMENT": 1, "I-MONUMENT": 2, "B-LOCATION": 3, "I-LOCATION": 4,
                          "<START>": 5, "<STOP>": 6, '<PAD>': 7} # Add a PAD tag for collate_fn

        # Save these mappings to be used by the inference script later
        vocab_map_path = os.path.join(self.output_dir, "word_to_ix.json")
        tag_map_path = os.path.join(self.output_dir, "tag_to_ix.json")
        with open(vocab_map_path, 'w') as f: json.dump(self.word_to_ix, f)
        with open(tag_map_path, 'w') as f: json.dump(self.tag_to_ix, f)

        print(f"  Vocabulary size: {len(self.word_to_ix)}")
        print(f"  Tag set size: {len(self.tag_to_ix)}")
        print(f"  Mappings saved to {self.output_dir}")

        # Update config to point to these newly created mapping files for inference
        self.ner_config['vocab_map_path'] = vocab_map_path
        self.ner_config['tag_map_path'] = tag_map_path


    def _load_data(self):
        """Loads training and validation datasets."""
        print("Loading NER training data...")
        train_dataset = NERDataset(
            data_file_path=self.train_config['dataset_path'],
            word_to_ix=self.word_to_ix,
            tag_to_ix=self.tag_to_ix
        )

        # For the collate function, we need to know the padding index for tags
        self.ner_collate_fn_with_padding = lambda batch: ner_collate_fn(batch) # Simplified
        # In a more robust implementation, pass padding_value to collate_fn if it's configurable

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_config.get('batch_size', 32),
            shuffle=True,
            collate_fn=self.ner_collate_fn_with_padding
        )
        # Add validation loader here if you have validation data
        self.val_loader = None
        print("Data loading complete.")

    def _build_model(self):
        """Builds the BiLSTM-CRF model."""
        print("Building BiLSTM-CRF model...")
        model_params = self.ner_config.get('model_params', {})
        self.model = BiLSTM_CRF(
            vocab_size=len(self.word_to_ix),
            tag_to_ix=self.tag_to_ix,
            embedding_dim=model_params.get('embedding_dim', 100),
            hidden_dim=model_params.get('hidden_dim', 256),
            dropout=model_params.get('dropout', 0.5)
        ).to(self.device)
        print("Model built successfully.")

    def _setup_optimizer(self):
        """Sets up the optimizer."""
        lr = self.train_config.get('learning_rate', 0.01)
        weight_decay = self.train_config.get('weight_decay', 1e-4)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"Optimizer: SGD, Learning Rate: {lr}")

    def train_epoch(self, epoch_num):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0

        for i, (sentences, tags, lengths) in enumerate(self.train_loader):
            sentences = sentences.to(self.device)
            tags = tags.to(self.device)

            self.model.zero_grad()

            # The loss function in BiLSTM-CRF is often calculated sentence by sentence
            # because CRF works on individual sequences. A batch-level CRF is more complex.
            # Our current model's `neg_log_likelihood` takes one sentence at a time.
            batch_loss = 0
            for j in range(sentences.size(0)): # Iterate over sentences in the batch
                sentence = sentences[j, :lengths[j]] # Get non-padded part
                tag_seq = tags[j, :lengths[j]]
                loss = self.model.neg_log_likelihood(sentence, tag_seq)
                batch_loss += loss

            # Average loss over the batch and perform backprop
            batch_loss /= sentences.size(0)
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()

            if (i + 1) % self.train_config.get('log_interval', 10) == 0:
                print(f"  Epoch {epoch_num+1}, Batch [{i+1}/{len(self.train_loader)}], Avg Batch Loss: {batch_loss.item():.4f}")

        avg_epoch_loss = total_loss / len(self.train_loader)
        print(f"End of Epoch {epoch_num+1}, Average Training Loss: {avg_epoch_loss:.4f}")

    def run_training(self):
        """Main training loop."""
        print("\n--- Starting NER Model Training ---")
        num_epochs = self.train_config.get('epochs', 10)

        for epoch in range(num_epochs):
            self.train_epoch(epoch)

            # Save model checkpoint
            if (epoch + 1) % self.train_config.get('save_epoch_interval', 1) == 0:
                save_path = os.path.join(self.output_dir, f"ner_model_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Model checkpoint saved to: {save_path}")

        # Update the main config file with the paths to the saved mappings for inference
        # This is a bit intrusive but helpful for a seamless workflow.
        self.config['ner']['vocab_map_path'] = self.ner_config['vocab_map_path']
        self.config['ner']['tag_map_path'] = self.ner_config['tag_map_path']
        self.config['ner']['model_path'] = os.path.join(self.output_dir, f"ner_model_epoch_{num_epochs}.pth") # Point to last model
        with open("configs/nlp.yaml", 'w') as f: # Overwrite the config file
            yaml.dump(self.config, f)
        print(f"Updated 'configs/nlp.yaml' with new model and mapping paths for inference.")

        print("\n--- NER Model Training Finished ---")

def main():
    parser = argparse.ArgumentParser(description="Train an NER model.")
    parser.add_argument('--config', type=str, default="configs/nlp.yaml",
                        help="Path to the NLP configuration YAML file.")
    args = parser.parse_args()

    # --- Dummy File Creation for Placeholder Run ---
    if not os.path.exists(args.config) or "temp_ner_data.conll" in open(args.config).read():
        print("Warning: Config file not found or is a dummy. Creating dummy files for NER training test.")
        dummy_dir = "temp_ner_train_files"
        os.makedirs(dummy_dir, exist_ok=True)

        dummy_data_path = os.path.join(dummy_dir, "temp_ner_data.conll")
        with open(dummy_data_path, 'w') as f: f.write("The\tO\nTaj\tB-MONUMENT\n\nAgra\tB-LOCATION\n")

        dummy_cfg = {
            'ner': {
                'training': {
                    'dataset_path': dummy_data_path,
                    'output_dir': 'models/nlp/ner/dummy_run/',
                    'epochs': 1,
                    'batch_size': 2,
                    'learning_rate': 0.01
                },
                'model_params': {'embedding_dim': 10, 'hidden_dim': 20}
            }
        }
        with open(args.config, 'w') as f: yaml.dump(dummy_cfg, f)

    # --- Main Execution ---
    try:
        trainer = NERTrainer(config_path=args.config)
        trainer.run_training()
    except Exception as e:
        print(f"\nAn error occurred during NER training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # --- Clean up dummy files ---
        with open(args.config, 'r') as f:
            content = f.read()
        if "temp_ner_data.conll" in content:
            print("\nCleaning up dummy files for NER training test...")
            cfg_data = yaml.safe_load(content)
            dataset_path = cfg_data.get('ner',{}).get('training',{}).get('dataset_path')
            if dataset_path and os.path.exists(dataset_path):
                os.remove(dataset_path)
            dummy_dir = os.path.dirname(dataset_path)
            if dummy_dir and os.path.exists(dummy_dir): os.rmdir(dummy_dir)


if __name__ == '__main__':
    # To run, you need a CoNLL-formatted dataset and a config file pointing to it.
    # The main function here creates dummy versions to allow a dry run.
    print("Executing src.nlp.train_ner...")
    main()
