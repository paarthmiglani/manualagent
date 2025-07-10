# src/ocr/postprocess.py
# Functions for postprocessing OCR model outputs

import numpy as np
# from typing import List, Tuple, Any # For type hinting

def ctc_decode_predictions(raw_preds, char_list, beam_search=False, beam_width=5, blank_idx=0):
    """
    Decodes raw model predictions (logits or probabilities) from a CTC-trained model.
    Args:
        raw_preds (np.ndarray or torch.Tensor): Model output, typically shape (SeqLen, NumClasses)
                                               or (Batch, SeqLen, NumClasses) if batched.
                                               Assumes probabilities (after softmax) or logits.
        char_list (list): List of characters corresponding to class indices.
                          The blank token should be part of this list or handled by blank_idx.
        beam_search (bool): Whether to use beam search decoding (more complex).
                            Simple best-path decoding is used if False.
        beam_width (int): Width of the beam if beam_search is True.
        blank_idx (int): Index of the blank token in char_list. Often 0.
    Returns:
        str: Decoded text string.
        float: Confidence score (can be basic, e.g., average probability of non-blank tokens).
    """
    print(f"CTC decoding raw predictions (placeholder)... Beam search: {beam_search}")

    # If raw_preds are logits, apply softmax
    # if not np.isclose(np.sum(raw_preds[0]), 1.0): # Simple check if not probabilities
    #    raw_preds = softmax(raw_preds, axis=-1) # Assuming softmax function is available

    # --- Best Path Decoding (Greedy) ---
    if not beam_search:
        # Get the character indices with the highest probability at each time step
        best_path_indices = np.argmax(raw_preds, axis=-1) # (SeqLen,)

        decoded_chars = []
        last_char_idx = -1 # To handle repeated characters separated by blank

        for char_idx in best_path_indices:
            if char_idx == blank_idx:
                last_char_idx = -1 # Reset, allows next char if same as previous non-blank
                continue
            if char_idx == last_char_idx: # Skip repeated non-blank characters
                continue

            if char_idx < len(char_list): # Ensure index is valid
                decoded_chars.append(char_list[char_idx])
            last_char_idx = char_idx

        recognized_text = "".join(decoded_chars)

        # Basic confidence: average probability of chosen characters (simplified)
        # This is a very naive confidence. More robust methods exist.
        confidence = 0.9 # Placeholder confidence
        # probs_of_best_path = np.max(raw_preds, axis=-1) # Probs of chosen char at each step
        # non_blank_probs = []
        # for i, char_idx in enumerate(best_path_indices):
        #     if char_idx != blank_idx and (i == 0 or char_idx != best_path_indices[i-1]):
        #         non_blank_probs.append(probs_of_best_path[i])
        # if non_blank_probs:
        #     confidence = np.mean(non_blank_probs)
        # else:
        #     confidence = 0.0

        print(f"Best path decoded text: '{recognized_text}', Confidence (dummy): {confidence:.2f}")
        return recognized_text, float(confidence)

    # --- Beam Search Decoding (Placeholder) ---
    else:
        print(f"Beam search decoding with beam_width={beam_width} (placeholder implementation)...")
        # Actual beam search is more involved, often using a language model too.
        # For a basic CTC beam search, you'd typically use libraries like ctcdecode
        # or implement the algorithm (e.g., prefix beam search).

        # Dummy output for beam search
        recognized_text = "BeamSearchDecodedText Placeholder"
        confidence = 0.85 # Placeholder confidence
        print(f"Beam search decoded text: '{recognized_text}', Confidence (dummy): {confidence:.2f}")
        return recognized_text, confidence


def aggregate_ocr_outputs(region_level_ocr_results, method="simple_join"):
    """
    Aggregates OCR results from multiple text regions into a coherent document-level text.
    Args:
        region_level_ocr_results (list of dicts):
            List of OCR outputs, e.g., from different detected text boxes.
            Each dict could contain:
                'text': recognized text in the region
                'bbox': [x_min, y_min, x_max, y_max] for the region
                'confidence': confidence of the region's OCR
                (Potentially 'line_num', 'block_num' if available from layout analysis)
        method (str): Aggregation method.
                      "simple_join": Joins text with spaces.
                      "reading_order": Attempts to sort by reading order (top-to-bottom, left-to-right).
    Returns:
        str: Aggregated text.
    """
    print(f"Aggregating {len(region_level_ocr_results)} OCR region outputs using method: '{method}' (placeholder)...")
    if not region_level_ocr_results:
        return ""

    if method == "reading_order":
        # Sort by y_min primarily, then x_min for tie-breaking (basic LTR, TTB)
        # More complex reading order might need layout analysis (e.g., column detection)
        # For simplicity, assume 'bbox' is [x_min, y_min, x_max, y_max]
        try:
            # Add a tolerance for y_min to group lines that are roughly at the same height
            # This is a simplified approach. True line grouping is more complex.
            y_tolerance = 10 # pixels
            sorted_results = sorted(
                region_level_ocr_results,
                key=lambda r: ( (r['bbox'][1] // y_tolerance) * y_tolerance, r['bbox'][0] )
            )
        except (KeyError, TypeError, IndexError) as e:
            print(f"Warning: Could not sort by reading order due to missing/malformed bbox: {e}. Falling back to simple join.")
            sorted_results = region_level_ocr_results # Fallback
    else: # "simple_join" or default
        sorted_results = region_level_ocr_results # No specific order, just join as is

    aggregated_text = " ".join([res.get('text', '') for res in sorted_results if res.get('text')])

    print(f"Aggregated text (first 100 chars): '{aggregated_text[:100]}...' (placeholder)")
    return aggregated_text.strip()


# Helper for softmax if not using a DL framework's built-in one
def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


if __name__ == '__main__':
    print("Testing OCR postprocessing functions (placeholders)...")

    # --- Test ctc_decode_predictions ---
    # Dummy raw_preds: (SeqLen, NumClasses)
    # SeqLen=10, NumClasses=5 (chars: blank, 'h', 'e', 'l', 'o')
    # char_list for this example: ['<blank>', 'h', 'e', 'l', 'o']
    dummy_char_list = ['<blank>', 'h', 'e', 'l', 'o']
    dummy_raw_preds_logits = np.array([
        [0.1, 0.8, 0.05, 0.05, 0.0], # h
        [0.1, 0.8, 0.05, 0.05, 0.0], # h (should be collapsed by CTC greedy)
        [0.9, 0.02, 0.02, 0.02, 0.04],# blank
        [0.05, 0.05, 0.8, 0.05, 0.05],# e
        [0.05, 0.05, 0.05, 0.8, 0.05],# l
        [0.05, 0.05, 0.05, 0.8, 0.05],# l (should be collapsed)
        [0.05, 0.05, 0.05, 0.8, 0.05],# l (still l)
        [0.9, 0.02, 0.02, 0.02, 0.04],# blank
        [0.05, 0.05, 0.05, 0.05, 0.8], # o
        [0.9, 0.02, 0.02, 0.02, 0.04] # blank
    ]) * 10 # Multiply to make logits more distinct before potential softmax

    # Apply softmax if your model outputs logits
    dummy_raw_preds_probs = softmax(dummy_raw_preds_logits, axis=-1)

    print("\nTesting Best Path Decoding:")
    text_greedy, conf_greedy = ctc_decode_predictions(dummy_raw_preds_probs, dummy_char_list, blank_idx=0)
    # Expected: "helo" (h h blank e l l l blank o blank -> h e l o)
    print(f"Greedy result: '{text_greedy}', Confidence (dummy): {conf_greedy}")
    assert text_greedy == "helo", f"Expected 'helo', got '{text_greedy}'"


    print("\nTesting Beam Search Decoding (placeholder):")
    text_beam, conf_beam = ctc_decode_predictions(dummy_raw_preds_probs, dummy_char_list, beam_search=True, beam_width=3, blank_idx=0)
    print(f"Beam search result: '{text_beam}', Confidence (dummy): {conf_beam}")


    # --- Test aggregate_ocr_outputs ---
    print("\nTesting OCR Aggregation:")
    dummy_regions = [
        {'text': 'second line', 'bbox': [10, 50, 200, 70], 'confidence': 0.9},
        {'text': 'first line', 'bbox': [10, 10, 180, 30], 'confidence': 0.92},
        {'text': 'word on second line', 'bbox': [210, 50, 350, 70], 'confidence': 0.88},
        {'text': 'third line, first part', 'bbox': [5, 90, 150, 110], 'confidence': 0.95},
        {'text': 'third line, second part', 'bbox': [160, 90, 300, 110], 'confidence': 0.93},
        {'text': '', 'bbox': [0,0,0,0]}, # Empty text region
        {'text': 'another box', 'bbox': [10, 10, 50, 20]} # Overlapping, should be ordered by reading order
    ]

    # Simple Join (order as is)
    agg_simple = aggregate_ocr_outputs(dummy_regions, method="simple_join")
    print(f"Aggregated (simple_join, placeholder): '{agg_simple}'")
    # Expected: "second line first line word on second line third line, first part third line, second part another box" (order dependent on list)

    # Reading Order
    agg_ordered = aggregate_ocr_outputs(dummy_regions, method="reading_order")
    print(f"Aggregated (reading_order, placeholder): '{agg_ordered}'")
    # Expected based on sorting logic (y then x for 'another box' vs 'first line'):
    # "another box first line second line word on second line third line, first part third line, second part"
    # The exact order of "another box" and "first line" depends on the y_tolerance and exact y values.
    # If 'another box' y1=10 and 'first line' y1=10, then 'another box' (x1=10) comes before 'first line' (x1=10, but it's identical so original order might be preserved for identical keys or it's unstable sort)
    # If y_tolerance groups them, then x sorting applies.
    # The dummy 'another box' has x1=10, y1=10. 'first line' has x1=10, y1=10.
    # So, their relative order might depend on Python's sort stability if primary keys are identical.
    # Let's assume stable sort: 'first line' then 'another box' if 'first line' appeared first in original list and keys are identical.
    # The sorting key `( (r['bbox'][1] // y_tolerance) * y_tolerance, r['bbox'][0] )`
    # For 'first line': ( (10//10)*10, 10 ) -> (10, 10)
    # For 'another box': ( (10//10)*10, 10 ) -> (10, 10)
    # Since keys are identical, stable sort preserves original relative order if they were not already sorted by this key.
    # Let's re-check the provided `dummy_regions` order:
    # 'second line' (y=50)
    # 'first line' (y=10)
    # 'word on second line' (y=50)
    # 'third line, first part' (y=90)
    # 'third line, second part' (y=90)
    # 'another box' (y=10)
    # After sort by (y_group, x):
    # (10,10) from 'first line'
    # (10,10) from 'another box'
    # (50,10) from 'second line'
    # (50,210) from 'word on second line'
    # (90,5) from 'third line, first part'
    # (90,160) from 'third line, second part'
    # If sort is stable, 'first line' comes before 'another box' because it was earlier in the list.
    # Expected: "first line another box second line word on second line third line, first part third line, second part"

    print("OCR postprocessing tests complete (placeholders).")
