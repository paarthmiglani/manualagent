# src/nlp/translation.py
# Implements the TextTranslator class for inference using the Transformer model.

import yaml
import torch
import os
import json

# Import model and utilities from our source files
from .model_definition_translation import Seq2SeqTransformer, generate_square_subsequent_mask
# Assuming a Vocabulary class will be available from the dataset file
# from .dataset_translation import Vocabulary

# --- Placeholder Vocabulary class until dataset_translation.py is fixed ---
class DummyVocabulary:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
    def encode(self, sentence):
        return [self.stoi.get(token, self.stoi['<UNK>']) for token in sentence.split()]
    def decode(self, indices):
        return " ".join([self.itos.get(str(ix), '<UNK>') for ix in indices])

class TextTranslator:
    def __init__(self, config_path, model_key):
        """
        Initializes the Text Translator for inference.
        Args:
            config_path (str): Path to the main NLP config file (e.g., configs/nlp.yaml).
            model_key (str): Key for a specific translation model (e.g., "en_hi").

        """
        with open(config_path, 'r') as f:
            nlp_config = yaml.safe_load(f)
        self.trans_config = nlp_config.get('translation', {}).get('models', {}).get(model_key)
        if not self.trans_config:
            raise ValueError(f"Translation config for model_key '{model_key}' not found.")

        self.infer_config = nlp_config.get('translation', {}).get('inference', {})
        self.model_key = model_key

        self.model = None
        self.src_vocab = None
        self.tgt_vocab = None
        self.device = None

        self._setup_device()
        self._load_vocab()
        self._load_model()
        print(f"TextTranslator for '{model_key}' initialized.")

    def _setup_device(self):
        self.device = torch.device(self.infer_config.get('device', 'cpu'))
        print(f"Using device: {self.device}")

    def _load_vocab(self):
        """Loads source and target vocabularies."""
        print("Loading translation vocabularies...")
        src_vocab_path = self.trans_config.get('src_vocab_path')
        tgt_vocab_path = self.trans_config.get('tgt_vocab_path')
        if not src_vocab_path or not tgt_vocab_path:
            raise ValueError("Paths to 'src_vocab_path' and 'tgt_vocab_path' must be specified in config.")

        # self.src_vocab = Vocabulary(file_path=src_vocab_path) # Assumes Vocabulary class handles loading
        # self.tgt_vocab = Vocabulary(file_path=tgt_vocab_path)
        # Placeholder loading:
        with open(src_vocab_path, 'r') as f: src_stoi = json.load(f)
        with open(tgt_vocab_path, 'r') as f: tgt_stoi = json.load(f)
        src_itos = {v: k for k, v in src_stoi.items()}
        tgt_itos = {v: k for k, v in tgt_stoi.items()}
        self.src_vocab = DummyVocabulary(src_stoi, src_itos)
        self.tgt_vocab = DummyVocabulary(tgt_stoi, tgt_itos)
        print("Vocabularies loaded.")

    def _load_model(self):
        """Loads the trained Transformer model."""
        model_path = self.trans_config.get('model_path')
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Translation model not found at: {model_path}")

        model_params = self.trans_config.get('model_params', {})
        self.model = Seq2SeqTransformer(
            num_encoder_layers=model_params['num_encoder_layers'],
            num_decoder_layers=model_params['num_decoder_layers'],
            emb_size=model_params['embedding_dim'],
            nhead=model_params['num_heads'],
            src_vocab_size=len(self.src_vocab.stoi),
            tgt_vocab_size=len(self.tgt_vocab.stoi),
            dim_feedforward=model_params.get('feedforward_dim', 512)
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("Translation model loaded successfully.")

    def translate(self, source_sentence, max_len=50):
        """
        Translates a single source sentence using greedy decoding.
        Args:
            source_sentence (str): The sentence to translate.
            max_len (int): The maximum length of the generated translation.
        Returns:
            str: The translated sentence.
        """
        self.model.eval()

        # 1. Preprocess the source sentence
        src_tokens = self.src_vocab.encode(source_sentence)
        src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(self.device) # (1, src_len)

        # 2. Create source masks
        src_padding_mask = (src_tensor == self.src_vocab.stoi['<PAD>']).to(self.device)

        # 3. Encode the source sentence
        with torch.no_grad():
            memory = self.model.encode(src_tensor, None, src_padding_mask)
        memory = memory.to(self.device)

        # 4. Greedy Decoding
        # Start with the <BOS> token
        tgt_indices = [self.tgt_vocab.stoi['<BOS>']]

        for i in range(max_len - 1):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(self.device)

            # Create target mask for decoder
            tgt_padding_mask = (tgt_tensor == self.tgt_vocab.stoi['<PAD>']).to(self.device)
            tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(1), self.device)

            with torch.no_grad():
                output = self.model.decode(tgt_tensor, memory, tgt_mask, tgt_padding_mask, None)
                # Get the logits for the last token
                last_token_logits = self.model.generator(output[:, -1])
                # Find the token with the highest probability
                pred_token_ix = last_token_logits.argmax(1).item()

            tgt_indices.append(pred_token_ix)

            # Stop if we predict the <EOS>
 