# src/nlp/dataset_translation.py
# Defines the custom Dataset and DataLoader logic for Translation.

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from collections import Counter

class Vocabulary:
    """
    Handles mapping between tokens and numerical indices.
    Can be built from a file or a list of sentences.
    """
    def __init__(self, special_tokens=['<UNK>', '<PAD>', '<BOS>', '<EOS>
