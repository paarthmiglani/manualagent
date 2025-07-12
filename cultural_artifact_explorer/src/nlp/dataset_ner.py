# src/nlp/dataset_ner.py
# Defines the custom Dataset and DataLoader logic for NER.

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

class NERDataset(Dataset):
    """
    Custom PyTorch Dataset for loading NER data, typically from a CoNLL-style format.
    A CoNLL file has one word and its corresponding tag per line, with empty lines
    separating sentences. Example:
    The     O
    Taj     B-MONUMENT
    Mahal   I-MONUMENT
    is      O
    in      O
    Agra    B-LOCATION
    .       O

    Another O
    sentence...
    """
    def __init__(self, data_file_path, word_to_ix, tag_to_ix):
        """
        Args:
            data_file_path (str): Path to the CoNLL-formatted data file.
            word_to_ix (dict): A dictionary mapping words to integer indices.
            tag_to_ix (dict): A dictionary mapping NER tags to integer indices.
        """
        super().__init__()
        self.sentences = []
        self.tags = []
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.unknown_word_ix = word_to_ix.get('<UNK>', 0) # Assume 0 is the <UNK> token index

        print(f"Initializing NERDataset from: {data_file_path}")

        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                current_sentence = []
                current_tags = []
                for line in f:
                    line = line.strip()
                    if line == "": # End of a sentence
                        if current_sentence:
                            self.sentences.append(current_sentence)
                            self.tags.append(current_tags)
                            current_sentence = []
                            current_tags = []
                    else:
                        # Expects tab-separated or space-separated word and tag
                        parts = line.split()
                        if len(parts) >= 2:
                            word, tag = parts[0], parts[-1] # Take first and last part
                            current_sentence.append(word)
                            current_tags.append(tag)

                # Add the last sentence if the file doesn't end with a newline
                if current_sentence:
                    self.sentences.append(current_sentence)
                    self.tags.append(current_tags)

            print(f"  Loaded {len(self.sentences)} sentences.")

        except Exception as e:
            print(f"Error loading or parsing NER data file {data_file_path}: {e}")
            # Dataset will be empty if file fails to load

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Returns one sample: a sentence and its corresponding tags, converted to indices.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        sentence = self.sentences[idx]
        tags = self.tags[idx]

        # Convert words to indices, using <UNK> for unknown words
        sentence_indices = [self.word_to_ix.get(w, self.unknown_word_ix) for w in sentence]

        # Convert tags to indices. Assume all tags in data file are in tag_to_ix.
        # If not, this will raise a KeyError, which is desirable to find data issues.
        try:
            tag_indices = [self.tag_to_ix[t] for t in tags]
        except KeyError as e:
            print(f"Error: Tag '{e.args[0]}' in sentence {idx} is not in the tag_to_ix mapping.")
            # Return the next valid item as a simple recovery strategy
            return self.__getitem__((idx + 1) % len(self)) if len(self) > 1 else (None, None)

        return torch.tensor(sentence_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)


def ner_collate_fn(batch):
    """
    Custom collate function for NER. Pads sentences and tags in a batch.
    Args:
        batch (list): A list of tuples, where each tuple is (sentence_tensor, tags_tensor).
    Returns:
        tuple: (padded_sentences, padded_tags, sentence_lengths)
    """
    # Filter out None samples from dataset errors
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    sentences, tags = zip(*batch)

    # Get the length of each sentence before padding
    sentence_lengths = torch.tensor([len(s) for s in sentences], dtype=torch.long)

    # Pad sentences and tags
    # The padding value should correspond to a <PAD> token in the vocabulary.
    # Let's assume word_to_ix['<PAD>'] is 0, or we can just use 0.
    # The model's embedding layer should be configured with `padding_idx=0`.
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)

    # For tags, the padding value doesn't matter as much if we use lengths
    # to mask the loss, but using a specific index can be cleaner.
    # Let's assume tag_to_ix['<PAD>'] exists, or just use 0.
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)

    return sentences_padded, tags_padded, sentence_lengths


if __name__ == '__main__':
    print("\n--- Testing NER Dataset and DataLoader ---")

    # Create dummy files for testing
    test_dir = "temp_ner_dataset_test"
    os.makedirs(test_dir, exist_ok=True)

    # Dummy data file
    data_path = os.path.join(test_dir, "dummy_data.conll")
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write("The\tO\n")
        f.write("Taj\tB-MONUMENT\n")
        f.write("Mahal\tI-MONUMENT\n")
        f.write("\n")
        f.write("Agra\tB-LOCATION\n")
        f.write("is\tO\n")
        f.write("a\tO\n")
        f.write("city\tO\n")

    # Dummy vocab and tag mappings
    # In a real pipeline, these would be built from the training data.
    word_to_ix = {'<PAD>': 0, '<UNK>': 1, 'The': 2, 'Taj': 3, 'Mahal': 4, 'Agra': 5, 'is': 6, 'a': 7, 'city': 8}
    tag_to_ix = {'<PAD>': 0, 'O': 1, 'B-MONUMENT': 2, 'I-MONUMENT': 3, 'B-LOCATION': 4}
    # Note: For BiLSTM-CRF, tag_to_ix must also include <START> and <STOP> tags.
    # This dataset class is agnostic to it; the training script would add them to the map.

    print("\n--- Initializing NERDataset ---")
    try:
        ner_dataset = NERDataset(
            data_file_path=data_path,
            word_to_ix=word_to_ix,
            tag_to_ix=tag_to_ix
        )
        print(f"Dataset size: {len(ner_dataset)}") # Expected: 2 sentences

        # Test __getitem__
        print("\n--- Testing dataset __getitem__ ---")
        sentence_sample, tags_sample = ner_dataset[0]
        print(f"Sample 0 Sentence (indices): {sentence_sample}")
        print(f"Sample 0 Tags (indices):    {tags_sample}")
        # Expected sentence: [2, 3, 4] -> The, Taj, Mahal
        # Expected tags: [1, 2, 3] -> O, B-MONUMENT, I-MONUMENT

        # Test DataLoader with collate_fn
        print("\n--- Testing DataLoader with ner_collate_fn ---")
        data_loader = DataLoader(ner_dataset, batch_size=2, shuffle=False, collate_fn=ner_collate_fn)

        batch = next(iter(data_loader))
        sentences_b, tags_b, lengths_b = batch

        print(f"Batch sentences shape: {sentences_b.shape}") # (B, MaxLen)
        print(f"Batch tags shape:      {tags_b.shape}")      # (B, MaxLen)
        print(f"Batch sentence lengths: {lengths_b}")       # (B,)

        # Expected shapes for this batch: sentences (2, 4), tags (2, 4), lengths [3, 4]
        if sentences_b.shape == (2, 4) and tags_b.shape == (2, 4) and torch.equal(lengths_b, torch.tensor([3, 4])):
            print("DataLoader test PASSED.")
        else:
            print("DataLoader test FAILED.")

    except Exception as e:
        print(f"An error occurred during NER dataset test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy files
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")

    print("\n--- NER Dataset Script Finished ---")
