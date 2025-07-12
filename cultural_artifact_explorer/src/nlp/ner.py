# src/nlp/ner.py

# Implements the NERTagger class for inference using the BiLSTM-CRF model.

import yaml
import json
import torch
import os

# Import model and utilities from our source files
from .model_definition_ner import BiLSTM_CRF
from .utils import preprocess_text_for_nlp # Optional preprocessing
class NERTagger:
    def __init__(self, config_path):
        """
        Initializes the NERTagger for inference.

        Args:
            config_path (str): Path to the main NLP config file (e.g., configs/nlp.yaml).
        """
        with open(config_path, 'r') as f:
            nlp_config = yaml.safe_load(f)

        self.ner_config = nlp_config.get('ner', {})
        if not self.ner_config:
            raise ValueError("NER configuration not found in NLP config.")

        self.model_path = self.ner_config.get('model_path')
        self.vocab_map_path = self.ner_config.get('vocab_map_path') # Path to word_to_ix.json
        self.tag_map_path = self.ner_config.get('tag_map_path') # Path to tag_to_ix.json

        self.model = None
        self.word_to_ix = None
        self.ix_to_tag = None
        self.device = None

        self._setup_device()
        self._load_mappings()
        self._load_model()
        print(f"NERTagger initialized. Model path: {self.model_path or 'Not set'}")

    def _setup_device(self):
        """Sets up the device for inference (CPU or GPU)."""
        device_str = self.ner_config.get('inference', {}).get('device', 'cpu')
        self.device = torch.device(device_str)
        print(f"Using device: {self.device}")

    def _load_mappings(self):
        """Loads word and tag mappings from files."""
        print("Loading NER vocabulary and tag mappings...")
        if not self.vocab_map_path or not self.tag_map_path:
            raise ValueError("Paths to 'vocab_map_path' and 'tag_map_path' must be specified in ner config.")

        try:
            with open(self.vocab_map_path, 'r', encoding='utf-8') as f:
                self.word_to_ix = json.load(f)
            with open(self.tag_map_path, 'r', encoding='utf-8') as f:
                self.tag_to_ix = json.load(f)

            # Create inverse mapping for tags to convert indices back to labels
            self.ix_to_tag = {i: tag for tag, i in self.tag_to_ix.items()}
            print(f"  Loaded vocabulary with {len(self.word_to_ix)} words.")
            print(f"  Loaded tag map with {len(self.tag_to_ix)} tags.")
        except Exception as e:
            print(f"Error loading mapping files: {e}")
            raise

    def _load_model(self):
        """Loads the trained BiLSTM-CRF model."""
        print(f"Loading NER model from: {self.model_path}...")
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"NER model not found at path: {self.model_path}")

        # Model parameters must match the saved model
        model_params = self.ner_config.get('model_params', {})
        embedding_dim = model_params.get('embedding_dim', 100) # Must match trained model
        hidden_dim = model_params.get('hidden_dim', 256)     # Must match trained model

        self.model = BiLSTM_CRF(
            vocab_size=len(self.word_to_ix),
            tag_to_ix=self.tag_to_ix,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            print("Ensure the model parameters in the config match the architecture of the saved model.")
            raise

        self.model.eval() # Set model to evaluation mode
        print("NER model loaded successfully.")

    def _preprocess_sentence(self, sentence_text):
        """Converts a raw sentence string to a tensor of word indices."""
        # Optional: Apply general text preprocessing like normalization
        # sentence_text = preprocess_text_for_nlp(sentence_text)

        # Tokenize (simple split for now; could be more advanced)
        words = sentence_text.split()

        # Convert words to indices
        unknown_ix = self.word_to_ix.get('<UNK>', 0)
        indices = [self.word_to_ix.get(w, unknown_ix) for w in words]

        return torch.tensor(indices, dtype=torch.long).to(self.device), words

    def extract_entities(self, text):
        """
        Extracts named entities from a given text sentence.
        Returns:
            list: A list of dictionaries for each found entity, e.g.,
                  [{'text': 'Taj Mahal', 'label': 'MONUMENT', 'start_char': 4, 'end_char': 13}]
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Cannot perform inference.")

        print(f"Extracting entities from text: \"{text[:50]}...\"")

        # Prepare sentence for the model
        sentence_tensor, original_tokens = self._preprocess_sentence(text)

        if sentence_tensor.nelement() == 0:
            return [] # Return empty list for empty input

        # Perform inference
        with torch.no_grad():
            score, tag_indices = self.model(sentence_tensor)

        # Convert tag indices back to string tags
        predicted_tags = [self.ix_to_tag[ix] for ix in tag_indices]

        # Aggregate tags into entities (handle B- and I- prefixes)
        return self._aggregate_tags_to_entities(original_tokens, predicted_tags, text)

    def _aggregate_tags_to_entities(self, tokens, tags, original_text):
        """Helper function to convert token-level BIO tags to character-level entity spans."""
        entities = []
        current_entity_tokens = []
        current_entity_label = None

        for i, (token, tag) in enumerate(zip(tokens, tags)):
            tag_prefix = tag.split('-')[0] if '-' in tag else 'O'
            tag_label = tag.split('-')[-1] if '-' in tag else None

            if tag_prefix == 'B':
                # If we were in the middle of another entity, save it first
                if current_entity_tokens:
                    entities.append(self._create_entity_dict(current_entity_tokens, current_entity_label, original_text))

                # Start a new entity
                current_entity_tokens = [token]
                current_entity_label = tag_label

            elif tag_prefix == 'I' and current_entity_label == tag_label:
                # Continue the current entity
                current_entity_tokens.append(token)

            else: # O-tag or a tag mismatch
                # End of any current entity
                if current_entity_tokens:
                    entities.append(self._create_entity_dict(current_entity_tokens, current_entity_label, original_text))

                # Reset
                current_entity_tokens = []
                current_entity_label = None

        # Add the last entity if the sentence ends with one
        if current_entity_tokens:
            entities.append(self._create_entity_dict(current_entity_tokens, current_entity_label, original_text))

        return entities

    def _create_entity_dict(self, tokens, label, original_text):
        """Creates the final dictionary for a found entity, including character offsets."""
        entity_text = " ".join(tokens)
        # Find start and end character indices. This is a simple search and can be brittle.
        # A more robust solution would use token offsets from a proper tokenizer.
        try:
            start_char = original_text.index(entity_text)
            end_char = start_char + len(entity_text)
        except ValueError:
            start_char, end_char = -1, -1 # Mark as not found if simple search fails

        return {
            'text': entity_text,
            'label': label,
            'start_char': start_char,
            'end_char': end_char
        }


def main():
    parser = argparse.ArgumentParser(description="Extract Named Entities from text.")
    parser.add_argument('--text', type=str, required=True, help="Text to process for NER.")
    parser.add_argument('--config', type=str, default="configs/nlp.yaml",
                        help="Path to the NLP configuration YAML file.")

    args = parser.parse_args()

    # --- Dummy File Creation for Placeholder Run ---
    # This setup allows the script to run end-to-end for demonstration.
    if not os.path.exists(args.config) or "temp_ner_vocab.json" in open(args.config).read():
        print("Warning: Config or dependent files not found. Creating dummy files for NER test run.")
        dummy_dir = "temp_ner_run_files"
        os.makedirs(dummy_dir, exist_ok=True)

        dummy_model_path = os.path.join(dummy_dir, "dummy_ner_model.pth")
        dummy_vocab_path = os.path.join(dummy_dir, "temp_ner_vocab.json")
        dummy_tag_map_path = os.path.join(dummy_dir, "temp_ner_tags.json")

        # Create dummy config
        dummy_cfg = {
            'ner': {
                'model_path': dummy_model_path,
                'vocab_map_path': dummy_vocab_path,
                'tag_map_path': dummy_tag_map_path,
                'model_params': {'embedding_dim': 10, 'hidden_dim': 20}, # Small params for dummy model
                'inference': {'device': 'cpu'}
            }
        }
        with open(args.config, 'w') as f: yaml.dump(dummy_cfg, f)

        # Create dummy mappings
        word_to_ix = {'<PAD>': 0, '<UNK>': 1, 'The': 2, 'Taj': 3, 'Mahal': 4, 'is': 5, 'in': 6, 'Agra': 7}
        tag_to_ix = {"O": 0, "B-MONUMENT": 1, "I-MONUMENT": 2, "B-LOCATION": 3, "I-LOCATION": 4, "<START>": 5, "<STOP>": 6}
        with open(dummy_vocab_path, 'w') as f: json.dump(word_to_ix, f)
        with open(dummy_tag_map_path, 'w') as f: json.dump(tag_to_ix, f)

        # Create dummy model with matching parameters
        dummy_model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, 10, 20)
        torch.save(dummy_model.state_dict(), dummy_model_path)
        print("Dummy model and mapping files created.")

    # --- Main Execution ---
    try:
        tagger = NERTagger(config_path=args.config)
        entities = tagger.extract_entities(args.text)

        print("\n--- NER Inference Results ---")
        print(f"  Input Text: {args.text}")
        print(f"  Extracted Entities:")
        if entities:
            print(json.dumps(entities, indent=2))
        else:
            print("  No entities found.")
        print("-----------------------------")

    except Exception as e:
        print(f"\nAn error occurred during NER inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # --- Clean up dummy files ---
        with open(args.config, 'r') as f:
            content = f.read()
        if "temp_ner_vocab.json" in content:
            print("\nCleaning up dummy files for NER test run...")
            cfg_data = yaml.safe_load(content)
            ner_cfg = cfg_data.get('ner', {})
            for path_key in ['model_path', 'vocab_map_path', 'tag_map_path']:
                if ner_cfg.get(path_key) and os.path.exists(ner_cfg[path_key]):
                    os.remove(ner_cfg[path_key])
            dummy_dir = os.path.dirname(ner_cfg.get('model_path'))
            if os.path.exists(dummy_dir): os.rmdir(dummy_dir)


if __name__ == '__main__':
    # Example: python src/nlp/ner.py --text "The Taj Mahal is in Agra" --config configs/nlp.yaml
    print("Executing src.nlp.ner (with implemented inference logic)...")
    main()
