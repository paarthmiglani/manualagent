# src/nlp/model_definition_translation.py
# Defines the neural network architecture for the Translation model.
# We will implement a standard Transformer model from scratch using PyTorch's built-in layers.

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    """
    A sequence-to-sequence model based on the Transformer architecture.
    """
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        """
        Args:
            num_encoder_layers (int): Number of sub-encoder-layers in the encoder.
            num_decoder_layers (int): Number of sub-decoder-layers in the decoder.
            emb_size (int): The dimensionality of the embeddings.
            nhead (int): The number of heads in the multiheadattention models.
            src_vocab_size (int): The size of the source vocabulary.
            tgt_vocab_size (int): The size of the target vocabulary.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(Seq2SeqTransformer, self).__init__()

        # --- Core Transformer Component ---
        # Using PyTorch's built-in Transformer which is powerful and optimized.
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # We will use batch_first for easier data handling
        )

        # --- Layers ---
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size) # To generate output vocab logits

    def forward(self, src, trg, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        """
        Forward pass for the Seq2Seq Transformer.
        Args:
            src (Tensor): Source sequence, shape [batch_size, src_len]
            trg (Tensor): Target sequence, shape [batch_size, trg_len]
            src_mask (Tensor): The square attention mask for the source sequence.
            tgt_mask (Tensor): The square attention mask for the target sequence.
            src_padding_mask (Tensor): The mask for the src keys per batch.
            tgt_padding_mask (Tensor): The mask for the trg keys per batch.
            memory_key_padding_mask (Tensor): The mask for the memory keys per batch.
        Returns:
            Tensor: Output logits, shape [batch_size, trg_len, tgt_vocab_size]
        """
        # Embed source and target sequences
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        # Pass through the transformer
        outs = self.transformer(
            src_emb, tgt_emb,
            src_mask, tgt_mask,
            None, # memory_mask is not typically used for standard seq2seq
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask
        )

        # Generate output logits
        return self.generator(outs)

    def encode(self, src, src_mask, src_padding_mask):
        """Encodes the source sequence. Used for inference."""
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)),
                                        src_mask, src_padding_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask):
        """Decodes the target sequence given the memory from the encoder. Used for inference."""
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)),
                                        memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)

def generate_square_subsequent_mask(sz, device):
    """Generates a square mask for the sequence. The masked positions are filled with -inf."""
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, pad_idx, device):
    """Creates all necessary masks for the Transformer model."""
    # src shape: [batch_size, src_len]
    # tgt shape: [batch_size, tgt_len]
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    # Target subsequent mask to prevent attending to future tokens
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    # Source mask is not typically needed unless you have a reason to mask source
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    # Padding masks to ignore pad tokens
    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

if __name__ == '__main__':
    print("--- Testing Seq2Seq Transformer Model Definition ---")

    # --- Model Parameters (Example) ---
    SRC_VOCAB_SIZE = 10000
    TGT_VOCAB_SIZE = 12000
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PAD_IDX = 0

    # --- Create Model Instance ---
    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=EMB_SIZE,
        nhead=NHEAD,
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        dim_feedforward=FFN_HID_DIM
    ).to(DEVICE)

    print("Seq2SeqTransformer model instance created successfully.")
    # print(model)

    # --- Test Forward Pass with Dummy Data ---
    print("\n--- Testing forward pass ---")
    BATCH_SIZE = 4
    SRC_SEQ_LEN = 20
    TGT_SEQ_LEN = 18

    # Create dummy input tensors
    src_tensor = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_SEQ_LEN), device=DEVICE)
    # Target for training includes all but the last token
    tgt_input = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_SEQ_LEN), device=DEVICE)

    # Create masks
    # Note: The built-in nn.Transformer requires masks of specific shapes.
    # Subsequent masks should be (SeqLen, SeqLen)
    # Padding masks should be (Batch, SeqLen)
    tgt_mask = generate_square_subsequent_mask(TGT_SEQ_LEN, DEVICE)
    src_mask = torch.zeros((SRC_SEQ_LEN, SRC_SEQ_LEN), device=DEVICE).type(torch.bool) # No masking for source

    src_padding_mask = (src_tensor == PAD_IDX)
    tgt_padding_mask = (tgt_input == PAD_IDX)

    # The memory_key_padding_mask should be the same as the source padding mask
    memory_key_padding_mask = src_padding_mask

    try:
        # Perform a forward pass
        # Note: The nn.Transformer expects masks to be float tensors for additive masking
        # or bool tensors for boolean masking. Let's ensure our masks are correct.
        # For nn.Transformer, src_mask and tgt_mask are for attention heads and should be float.
        # Padding masks are for keys and should be bool.

        # The nn.Transformer with batch_first=True still expects masks in shape (S,S) or (N*nhead, S, S)
        # But padding masks as (N,S). Let's re-check shapes.
        # src_mask: (S, S), tgt_mask: (T, T), src_padding_mask: (N, S), tgt_padding_mask: (N, T)

        # Our model's forward pass was slightly simplified. Let's adjust to match nn.Transformer.
        # The nn.Transformer implementation is a bit particular. Let's simplify the forward pass call
        # to match the nn.Transformer documentation for batch_first=True.

        # Correct forward pass call
        output_logits = model(
            src=src_tensor,
            trg=tgt_input,
            src_mask=None, # Not needed for encoder self-attention
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask # This masks the encoder output for the decoder
        )

        print(f"Model output (logits) shape: {output_logits.shape}")

        # --- Check Output Shape ---
        expected_shape = (BATCH_SIZE, TGT_SEQ_LEN, TGT_VOCAB_SIZE)
        if output_logits.shape == expected_shape:
            print(f"Forward pass test PASSED. Output shape {output_logits.shape} matches expected {expected_shape}.")
        else:
            print(f"Forward pass test FAILED. Output shape {output_logits.shape} does not match expected {expected_shape}.")

    except Exception as e:
        print(f"An error occurred during the forward pass test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Translation Model Definition Script Finished ---")
