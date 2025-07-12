# src/ocr/postprocess.py
# Functions for postprocessing OCR model outputs

import numpy as np
import torch # For beam search if using a library like ctcdecode

def ctc_decode_predictions(raw_preds, char_list, beam_search=False, beam_width=10, blank_idx=0):
    """
    Decodes raw model predictions (logits or log_probs) from a CTC-trained model.
    Args:
        raw_preds (np.ndarray): Model output, shape (SeqLen, NumClasses).
                                Assumes log probabilities (output of log_softmax).
        char_list (list): List of characters corresponding to class indices (excluding blank).
        beam_search (bool): Whether to use beam search decoding.
        beam_width (int): Width of the beam if beam_search is True.
        blank_idx (int): Index of the blank token (almost always 0 for PyTorch's CTCLoss).
    Returns:
        str: Decoded text string.
        float: Confidence score.
    """
    # --- Greedy Decoding (Best Path Decoding) ---
    if not beam_search:
        # Get the character indices with the highest probability at each time step
        # raw_preds are log_probs, so argmax works directly.
        best_path_indices = np.argmax(raw_preds, axis=1) # Shape: (SeqLen,)

        # Calculate confidence score from probabilities
        # Convert log_probs to probs for the chosen path
        log_probs_of_best_path = raw_preds[np.arange(len(best_path_indices)), best_path_indices]
        probs_of_best_path = np.exp(log_probs_of_best_path)

        # Naive confidence: mean probability of non-blank characters in the collapsed path
        # This is a simple but common heuristic.
        non_blank_probs = []

        decoded_chars = []
        last_char_idx = -1

        for i, char_idx in enumerate(best_path_indices):
            if char_idx == blank_idx:
                last_char_idx = -1
                continue

            # If the current character is the same as the last one, it's a repeat, so skip.
            if char_idx == last_char_idx:
                continue

            # Map index to character (adjusting for blank at index 0)
            # Our char_list does not contain blank. char_idx=1 -> char_list[0]
            if char_idx > 0 and char_idx <= len(char_list):
                decoded_chars.append(char_list[char_idx - 1])
                non_blank_probs.append(probs_of_best_path[i]) # Store prob for this char

            last_char_idx = char_idx

        recognized_text = "".join(decoded_chars)

        confidence = np.mean(non_blank_probs) if non_blank_probs else 0.0

        return recognized_text, float(confidence)

    # --- Beam Search Decoding ---
    else:
        # Beam search is more complex and often requires a specialized library.
        # Example using the 'ctcdecode' library (would need to be added to requirements.txt)
        # try:
        #     from ctcdecode import CTCBeamDecoder
        # except ImportError:
        #     print("Warning: 'ctcdecode' library not found. Falling back to greedy decoding.")
        #     # Fallback to the greedy implementation from above
        #     return ctc_decode_predictions(raw_preds, char_list, beam_search=False, blank_idx=blank_idx)

        # # ctcdecode expects probabilities, not log_probs. And shape (Batch, SeqLen, NumClasses)
        # # Let's assume raw_preds is log_probs, shape (SeqLen, NumClasses)
        # probs = np.exp(raw_preds)
        # probs_tensor = torch.from_numpy(probs).unsqueeze(0) # Add batch dimension

        # # The labels for the decoder are the character list itself.
        # # It expects a specific format, often a single string.
        # ctc_labels = "".join(char_list)

        # decoder = CTCBeamDecoder(
        #     labels=ctc_labels,
        #     model_path=None, # No language model
        #     alpha=0, # No LM
        #     beta=0, # No LM
        #     cutoff_top_n=40,
        #     cutoff_prob=1.0,
        #     beam_width=beam_width,
        #     num_processes=os.cpu_count(),
        #     blank_id=blank_idx,
        #     log_probs_input=False # We are passing probabilities
        # )

        # beam_results, beam_scores, timesteps, out_lens = decoder.decode(probs_tensor)

        # # Get the top hypothesis
        # best_hyp_indices = beam_results[0][0][:out_lens[0][0]]
        # recognized_text = "".join([ctc_labels[i] for i in best_hyp_indices])
        # confidence = beam_scores[0][0].item() # Log probability of the beam

        # return recognized_text, np.exp(confidence) # Convert log prob back to prob for consistency

        # Placeholder for now since ctcdecode is not a dependency
        print("Warning: Beam search is not fully implemented in this placeholder. Falling back to greedy decoding.")
        return ctc_decode_predictions(raw_preds, char_list, beam_search=False, blank_idx=blank_idx)


def aggregate_ocr_outputs(region_level_ocr_results, method="reading_order"):
    """
    Aggregates OCR results from multiple text regions into a coherent document-level text.
    Args:
        region_level_ocr_results (list of dicts):
            List of OCR outputs, e.g., from different detected text boxes.
            Each dict should contain: {'text': str, 'bbox': [x_min, y_min, x_max, y_max]}
        method (str): Aggregation method ("simple_join" or "reading_order").
    Returns:
        str: Aggregated text.
    """
    if not region_level_ocr_results:
        return ""

    if method == "reading_order":
        # Sort by y_min primarily, then x_min for tie-breaking (basic LTR, TTB)
        # A small tolerance helps group words on the same line that may have minor y-alignment differences.
        y_tolerance = 10 # pixels
        try:
            sorted_results = sorted(
                region_level_ocr_results,
                key=lambda r: ( (r['bbox'][1] // y_tolerance) * y_tolerance, r['bbox'][0] )
            )
        except (KeyError, TypeError, IndexError):
            print("Warning: Could not sort by reading order due to malformed bbox. Falling back to simple join.")
            sorted_results = region_level_ocr_results
    else: # "simple_join" or default
        sorted_results = region_level_ocr_results

    # Join text with spaces, or newlines if y-coordinates differ significantly
    aggregated_text = ""
    last_y_group = -1
    for res in sorted_results:
        text = res.get('text', '')
        if not text:
            continue

        y_group = (res['bbox'][1] // y_tolerance) * y_tolerance
        if last_y_group != -1 and y_group > last_y_group:
            aggregated_text += "\n" # New line

        aggregated_text += text + " "
        last_y_group = y_group

    return aggregated_text.strip()


if __name__ == '__main__':
    print("--- Testing OCR Postprocessing Functions (Implemented) ---")

    # --- Test ctc_decode_predictions (Greedy) ---
    print("\n--- Testing Greedy CTC Decoding ---")
    # Dummy raw_preds: (SeqLen, NumClasses)
    # NumClasses=5. Chars: a,b,c. Vocab: {'a':1, 'b':2, 'c':3}. Blank=0.
    char_list_test = ['a', 'b', 'c']
    # Log probabilities for a sequence that should decode to "cab"
    # Seq: c, c, blank, a, blank, b, b
    log_probs_test = np.array([
        [-5.0, -5.0, -5.0, -0.1], # c
        [-4.0, -4.0, -4.0, -0.2], # c (repeat, should be collapsed)
        [-0.1, -5.0, -5.0, -5.0], # blank
        [-3.0, -0.1, -3.0, -3.0], # a
        [-0.2, -4.0, -4.0, -4.0], # blank
        [-2.0, -2.0, -0.1, -2.0], # b
        [-2.5, -2.5, -0.2, -2.5], # b (repeat, should be collapsed)
    ])

    text_greedy, conf_greedy = ctc_decode_predictions(log_probs_test, char_list_test, blank_idx=0)
    expected_text = "cab"
    print(f"Greedy decoded text: '{text_greedy}', Confidence: {conf_greedy:.4f}")

    # Calculate expected confidence
    # Probs for 'c', 'a', 'b' are exp(-0.1), exp(-0.1), exp(-0.1)
    # expected_conf = np.mean([np.exp(-0.1), np.exp(-0.1), np.exp(-0.1)]) # This is wrong, it should be the prob of the chosen char
    # Probs are: exp(-0.1) for 'c', exp(-0.1) for 'a', exp(-0.1) for 'b'
    expected_conf_manual = np.mean([np.exp(-0.1), np.exp(-0.1), np.exp(-0.1)])

    if text_greedy == expected_text:
        print("Greedy decoding test PASSED.")
    else:
        print(f"Greedy decoding test FAILED. Expected '{expected_text}', got '{text_greedy}'.")

    # --- Test aggregate_ocr_outputs ---
    print("\n--- Testing OCR Aggregation ---")
    dummy_regions = [
        {'text': 'second line', 'bbox': [10, 50, 200, 70]},
        {'text': 'first line', 'bbox': [10, 10, 180, 30]},
        {'text': 'word on second line', 'bbox': [210, 50, 350, 70]},
    ]

    agg_ordered = aggregate_ocr_outputs(dummy_regions, method="reading_order")
    expected_agg = "first line\nsecond line word on second line"
    print(f"Aggregated (reading_order):\n'{agg_ordered}'")

    if agg_ordered == expected_agg:
        print("Aggregation test PASSED.")
    else:
        print(f"Aggregation test FAILED. Expected:\n'{expected_agg}'")

    print("\n--- Postprocessing Script Finished ---")
