# src/nlp/model_definition_ner.py
# Defines the neural network architecture for the Named Entity Recognition (NER) model.
# We will implement a BiLSTM-CRF model, a classic and strong baseline for sequence tagging.

import torch
import torch.nn as nn

class BiLSTM_CRF(nn.Module):
    """
    BiLSTM-CRF model for Named Entity Recognition.
    This model consists of:
    1. An embedding layer to convert words to dense vectors.
    2. A bidirectional LSTM (BiLSTM) to capture context from both directions.
    3. A linear layer to project LSTM outputs to the tag space.
    4. A Conditional Random Field (CRF) layer to model dependencies between tags
       and predict the most likely sequence of tags.
    """
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout=0.5):
        """
        Initializes the BiLSTM-CRF model.
        Args:
            vocab_size (int): The number of words in the vocabulary.
            tag_to_ix (dict): A dictionary mapping NER tags to integer indices.
            embedding_dim (int): The dimensionality of word embeddings.
            hidden_dim (int): The dimensionality of the LSTM hidden state.
            dropout (float): Dropout probability.
        """
        super(BiLSTM_CRF, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # 1. Embedding Layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        # 2. BiLSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        # 3. Linear layer that maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 4. CRF Layer
        # The CRF layer will learn transition probabilities between tags.
        # We add two special tags: <START> and <STOP> for the CRF.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # We never transition to the start tag, and we never transition from the stop tag.
        # We can enforce these constraints by setting their transition scores to a very low number.
        self.transitions.data[tag_to_ix['<START>'], :] = -10000
        self.transitions.data[:, tag_to_ix['<STOP>']] = -10000

    def _get_lstm_features(self, sentence):
        """Gets the emission scores from the BiLSTM."""
        # sentence shape: (batch_size, seq_len)
        embeds = self.word_embeds(sentence) # -> (batch_size, seq_len, embedding_dim)
        embeds = self.dropout(embeds)

        lstm_out, _ = self.lstm(embeds) # -> (batch_size, seq_len, hidden_dim)
        lstm_out = self.dropout(lstm_out)

        lstm_feats = self.hidden2tag(lstm_out) # -> (batch_size, seq_len, tagset_size)
        return lstm_feats

    def _forward_alg(self, feats):
        """
        Calculates the partition function (log sum of all possible tag sequence scores)
        using the forward algorithm.
        """
        # feats shape: (seq_len, tagset_size)
        init_alphas = torch.full((1, self.tagset_size), -10000., device=feats.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix['<START>']] = 0.

        forward_var = init_alphas

        for feat in feats:
            # feat shape: (tagset_size,)
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix['<STOP>']]
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _score_sentence(self, feats, tags):
        """Calculates the score of a given tag sequence."""
        # feats shape: (seq_len, tagset_size)
        # tags shape: (seq_len,)
        score = torch.zeros(1, device=feats.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix['<START>']], dtype=torch.long, device=tags.device), tags])

        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix['<STOP>'], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        """Finds the most likely tag sequence using the Viterbi algorithm."""
        # feats shape: (seq_len, tagset_size)
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000., device=feats.device)
        init_vvars[0][self.tag_to_ix['<START>']] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix['<STOP>']]
        best_tag_id = torch.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_ix['<START>']
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        """
        Calculates the negative log-likelihood loss.
        This is what will be minimized during training.
        """
        # This function processes a single sentence/tags pair.
        # It needs to be called in a loop for a batch.
        feats = self._get_lstm_features(sentence.unsqueeze(0)).squeeze(0) # Process one sentence
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        """
        Inference method: predicts the tag sequence for a given sentence.
        """
        # This function processes a single sentence.
        # For batch prediction, this would need to be looped or redesigned.
        lstm_feats = self._get_lstm_features(sentence.unsqueeze(0)).squeeze(0)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


if __name__ == '__main__':
    print("--- Testing BiLSTM-CRF Model Definition ---")

    # --- Model Parameters (Example) ---
    VOCAB_SIZE = 1000
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 128
    # Example tag mapping. In a real scenario, this would be loaded from a file.
    # It must include <START> and <STOP> tags for the CRF.
    TAG_TO_IX = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "<START>": 5, "<STOP>": 6}

    # --- Create Model Instance ---
    model = BiLSTM_CRF(
        vocab_size=VOCAB_SIZE,
        tag_to_ix=TAG_TO_IX,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    )
    print("BiLSTM-CRF model instance created successfully.")
    # print(model)

    # --- Test Forward Pass and Loss Calculation (Inference and Training) ---
    print("\n--- Testing forward pass and loss calculation ---")

    # Create dummy input data for one sentence
    # In a real case, words are mapped to integers from the vocabulary.
    sentence_in = torch.randint(0, VOCAB_SIZE, (15,)) # Sentence of length 15
    tags_in = torch.tensor([0, 1, 2, 0, 3, 4, 4, 0], dtype=torch.long) # Ground truth tags for a shorter sentence
    sentence_in_short = sentence_in[:8]

    try:
        # 1. Test Loss Calculation (what you'd do in a training loop)
        print("Testing neg_log_likelihood (for training)...")
        loss = model.neg_log_likelihood(sentence_in_short, tags_in)
        print(f"Calculated loss (dummy data): {loss.item():.4f}")
        # The loss should be a positive value.
        if loss.item() >= 0:
            print("Loss calculation test PASSED.")
        else:
            print("Loss calculation test FAILED. Loss should be non-negative.")

        # 2. Test Inference/Forward Pass (what you'd do for prediction)
        print("\nTesting forward() for inference...")
        with torch.no_grad(): # No need to calculate gradients for inference
            score, tag_sequence = model(sentence_in)

        print(f"Inference score (dummy data): {score.item():.4f}")
        print(f"Predicted tag sequence (indices): {tag_sequence}")
        # The length of the predicted sequence should match the input sentence length.
        if len(tag_sequence) == len(sentence_in):
            print("Inference output length test PASSED.")
        else:
             print(f"Inference output length test FAILED. Expected {len(sentence_in)}, got {len(tag_sequence)}.")

    except Exception as e:
        print(f"An error occurred during the model test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- NER Model Definition Script Finished ---")
