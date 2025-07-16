# src/ocr/postprocess.py
# Postprocessing for CRNN+CTC: decoding logits to text, removing duplicates/blanks.

import numpy as np

def ctc_decode_predictions(log_probs, char_list, blank_idx=0):
    """
    Decodes the output of the network with CTC: argmax, remove blanks/repeats.
    Args:
        log_probs: np.ndarray, shape (T, C) or (T, N, C) or (N, T, C) (softmax/argmax already done)
        char_list: list of str (should have <BLANK> at index blank_idx)
        blank_idx: int
    Returns:
        decoded_text: str
        raw_indices: list[int]
    """
    # Accepts (T, C) or (T, N, C) or (N, T, C)
    # We'll assume (T, C) for a single example
    if len(log_probs.shape) == 3:
        # (T, N, C) or (N, T, C)
        if log_probs.shape[1] == len(char_list):
            # (T, C, N)
            log_probs = log_probs.transpose(2, 0, 1)
        # We'll process batch element-wise externally
        raise ValueError("ctc_decode_predictions expects (T, C) array for a single sample.")

    # 1. Take argmax over classes at each time step
    pred_indices = np.argmax(log_probs, axis=1)  # (T,)

    # 2. Remove duplicate indices, and blanks
    decoded = []
    prev_idx = None
    for idx in pred_indices:
        if idx != prev_idx and idx != blank_idx:
            decoded.append(idx)
        prev_idx = idx
    decoded_text = "".join([char_list[i] for i in decoded if i < len(char_list) and i != blank_idx])
    return decoded_text, decoded

# If you want beam search, you can use third-party CTC beam search decoders (optional).

if __name__ == "__main__":
    # Simple test
    dummy_logits = np.array([
        [1.0, 2.0, 0.2],
        [1.1, 2.1, 0.1],
        [1.5, 0.1, 3.0],
        [2.5, 1.1, 0.1],
    ])  # (T=4, C=3)
    chars = ["<BLANK>", "a", "b"]
    text, indices = ctc_decode_predictions(dummy_logits, chars, blank_idx=0)
    print(f"Decoded: {text} / indices: {indices}")
