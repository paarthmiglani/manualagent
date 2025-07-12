# src/utils/metrics.py
# Utility functions for calculating various evaluation metrics.
# Renamed from 'metrics.py' to avoid potential naming conflicts if a 'metrics' sub-package were created.

import numpy as np
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# For NLP specific metrics:
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from sacrebleu import corpus_bleu # Often preferred for standardized BLEU
# For NER metrics:
# from seqeval.metrics import classification_report as ner_classification_report
# from seqeval.scheme import IOB2 # Or other tagging schemes


# --- General Classification Metrics ---

def calculate_accuracy(y_true, y_pred):
    """Calculates accuracy.
    Args:
        y_true (list or np.array): True labels.
        y_pred (list or np.array): Predicted labels.
    Returns:
        float: Accuracy score.
    """
    print("Calculating accuracy (placeholder in metrics.py)...")
    # return accuracy_score(y_true, y_pred)
    if len(y_true) == 0: return 0.0
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true) # Placeholder calculation

def calculate_precision_recall_f1(y_true, y_pred, average='weighted', labels=None):
    """
    Calculates precision, recall, and F1-score.
    Args:
        y_true (list or np.array): True labels.
        y_pred (list or np.array): Predicted labels.
        average (str): Type of averaging for multiclass ('micro', 'macro', 'weighted', 'samples', None).
        labels (list, optional): The set of labels to include when average is not None.
    Returns:
        dict: {'precision': float, 'recall': float, 'f1_score': float, 'support': (optional)}
    """
    print(f"Calculating P/R/F1 (average: {average}) (placeholder in metrics.py)...")
    # p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=average, labels=labels, zero_division=0)
    # return {'precision': p, 'recall': r, 'f1_score': f1, 'support': s}
    # Placeholder calculation (binary/macro-like for simplicity)
    # This is a highly simplified placeholder and not a correct general P/R/F1 calculation.
    if len(y_true) == 0: return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    # Assume positive class is 1 for a simple binary placeholder
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1_score': f1}


# --- NLP Specific Metrics ---

def calculate_bleu_score(references, hypotheses, smoothing_function=None):
    """
    Calculates BLEU score for machine translation or text generation.
    Args:
        references (list of list of str): List of reference translations,
                                          each reference is a list of tokens.
                                          Example: [[['this', 'is', 'a', 'test']], [['this', 'is', 'the', 'test']]]
                                          (Outer list for multiple candidate sentences, inner for multiple references per candidate)
        hypotheses (list of list of str): List of hypothesis translations,
                                          each hypothesis is a list of tokens.
                                          Example: [['this', 'is', 'a', 'test'], ['this', 'is', 'my', 'test']]
        smoothing_function: NLTK smoothing function (e.g., SmoothingFunction().method1).
    Returns:
        float: Corpus BLEU score (placeholder, could be sentence average).
    """
    print("Calculating BLEU score (placeholder in metrics.py)...")
    # if not references or not hypotheses or len(references) != len(hypotheses):
    #     print("Warning: Invalid input for BLEU calculation. Returning 0.")
    #     return 0.0

    # Using sacrebleu is often preferred for comparable scores:
    # hypotheses_str = [" ".join(h) for h in hypotheses]
    # references_str_grouped = [[" ".join(ref) for ref in group] for group in references] # if multiple refs per hypo
    # bleu = corpus_bleu(hypotheses_str, references_str_grouped)
    # return bleu.score / 100.0 # sacrebleu score is 0-100

    # Simple NLTK-like placeholder (averaging sentence BLEU)
    # total_bleu = 0
    # count = 0
    # for refs_for_one_hypo, hypo in zip(references, hypotheses):
    #     try:
    #         # sent_bleu = sentence_bleu(refs_for_one_hypo, hypo, smoothing_function=smoothing_function or SmoothingFunction().method1)
    #         # total_bleu += sent_bleu
    #         # count +=1
    #         pass # Placeholder for NLTK call
    #     except Exception as e:
    #         print(f"  Warning: Error calculating sentence BLEU: {e}")
    # if count == 0: return 0.0
    # return total_bleu / count
    return np.random.uniform(0.1, 0.5) # Placeholder: random BLEU score


def calculate_ner_metrics(y_true_tags, y_pred_tags, scheme='IOB2'):
    """
    Calculates precision, recall, F1 for NER tasks using seqeval.
    Args:
        y_true_tags (list of list of str): True NER tags for each token in each sentence.
                                           Example: [['O', 'B-PER', 'I-PER', 'O'], ['B-LOC', 'O']]
        y_pred_tags (list of list of str): Predicted NER tags.
        scheme (str): Tagging scheme (e.g., 'IOB2', 'IOBES').
    Returns:
        dict: Classification report from seqeval (or placeholder).
    """
    print(f"Calculating NER metrics (scheme: {scheme}) (placeholder in metrics.py)...")
    # try:
    #     # report = ner_classification_report(y_true_tags, y_pred_tags, scheme=scheme, output_dict=True, zero_division=0)
    #     # return report
    # except Exception as e:
    #     print(f"Error calculating NER metrics with seqeval: {e}. Returning dummy report.")
    # Placeholder report
    return {
        'PERSON': {'precision': np.random.uniform(0.5,0.9), 'recall': np.random.uniform(0.5,0.9), 'f1-score': np.random.uniform(0.5,0.9), 'support': np.random.randint(5,20)},
        'LOCATION': {'precision': np.random.uniform(0.5,0.9), 'recall': np.random.uniform(0.5,0.9), 'f1-score': np.random.uniform(0.5,0.9), 'support': np.random.randint(5,20)},
        'micro avg': {'precision': np.random.uniform(0.5,0.9), 'recall': np.random.uniform(0.5,0.9), 'f1-score': np.random.uniform(0.5,0.9), 'support': np.random.randint(10,40)},
        'macro avg': {'precision': np.random.uniform(0.5,0.9), 'recall': np.random.uniform(0.5,0.9), 'f1-score': np.random.uniform(0.5,0.9), 'support': np.random.randint(10,40)},
    }


# --- OCR Specific Metrics ---
def calculate_cer(reference_text, hypothesis_text):
    """
    Calculates Character Error Rate (CER).
    CER = (Substitutions + Insertions + Deletions) / Number of characters in reference.
    Requires an alignment algorithm (e.g., Levenshtein distance).
    """
    print("Calculating CER (placeholder in metrics.py)...")
    # if not isinstance(reference_text, str) or not isinstance(hypothesis_text, str):
    #     return 1.0 # Max error if inputs are invalid
    # if not reference_text: # Empty reference
    #     return 1.0 if hypothesis_text else 0.0 # All insertions if hypothesis not empty

    # # Using a library like 'editdistance' is common
    # # import editdistance
    # # distance = editdistance.eval(reference_text, hypothesis_text)
    # # cer = distance / len(reference_text)
    # # return cer

    # Basic Levenshtein distance placeholder (can be slow for long strings)
    if reference_text == hypothesis_text: return 0.0
    # This is a very simplified placeholder, real Levenshtein is more complex.
    # For simplicity, just return a random error rate.
    return np.random.uniform(0.05, 0.3) if reference_text else 1.0


def calculate_wer(reference_text, hypothesis_text):
    """
    Calculates Word Error Rate (WER).
    WER = (Substitutions + Insertions + Deletions) / Number of words in reference.
    Requires word tokenization and alignment.
    """
    print("Calculating WER (placeholder in metrics.py)...")
    # if not isinstance(reference_text, str) or not isinstance(hypothesis_text, str):
    #     return 1.0

    # ref_words = reference_text.split()
    # hyp_words = hypothesis_text.split()

    # if not ref_words:
    #     return 1.0 if hyp_words else 0.0

    # # Using a library like 'editdistance' on word lists
    # # import editdistance
    # # distance = editdistance.eval(ref_words, hyp_words) # Compares lists element by element
    # # wer = distance / len(ref_words)
    # # return wer
    return np.random.uniform(0.1, 0.4) if reference_text.split() else 1.0


if __name__ == '__main__':
    print("Testing metrics utility functions (placeholders)...")

    # --- Test Classification Metrics ---
    print("\n--- Testing Classification Metrics ---")
    y_true_class = [0, 1, 1, 0, 1, 0, 0, 1]
    y_pred_class = [0, 1, 0, 0, 1, 1, 0, 1]
    acc = calculate_accuracy(y_true_class, y_pred_class)
    print(f"Accuracy (dummy): {acc:.4f}") # Expected for this data: 6/8 = 0.75

    # For P/R/F1, the placeholder is very basic (binary, positive class 1)
    # Real test would need scikit-learn or a proper implementation.
    prf1 = calculate_precision_recall_f1(y_true_class, y_pred_class)
    print(f"Precision/Recall/F1 (dummy for class 1): {prf1}")
    # TP=3 (idx 1,4,7), FP=1 (idx 5), FN=1 (idx 2)
    # Prec = 3/(3+1)=0.75, Rec=3/(3+1)=0.75, F1=0.75

    # --- Test NLP Metrics ---
    print("\n--- Testing NLP Metrics ---")
    # BLEU Score
    refs_bleu = [[['this', 'is', 'a', 'test'], ['this', 'is', 'the', 'test']]] # One sentence, two references
    hyps_bleu = [['this', 'is', 'a', 'test']]
    bleu = calculate_bleu_score(refs_bleu, hyps_bleu)
    print(f"BLEU Score (dummy): {bleu:.4f}") # Perfect match should be 1.0 with NLTK, placeholder is random

    # NER Metrics
    y_true_ner = [['O', 'B-PER', 'I-PER', 'O'], ['B-LOC', 'O']]
    y_pred_ner = [['O', 'B-PER', 'O', 'O'], ['B-LOC', 'O']] # Mistake on I-PER
    ner_report = calculate_ner_metrics(y_true_ner, y_pred_ner)
    print(f"NER Report (dummy):\n{json.dumps(ner_report, indent=2)}")

    # --- Test OCR Metrics ---
    print("\n--- Testing OCR Metrics ---")
    ref_ocr = "hello world example"
    hyp_ocr_good = "hello world example"
    hyp_ocr_bad = "helo word exampel"

    cer_good = calculate_cer(ref_ocr, hyp_ocr_good)
    print(f"CER (good, dummy): {cer_good:.4f}") # Expected 0.0
    cer_bad = calculate_cer(ref_ocr, hyp_ocr_bad)
    print(f"CER (bad, dummy): {cer_bad:.4f}") # Expected > 0.0

    wer_good = calculate_wer(ref_ocr, hyp_ocr_good)
    print(f"WER (good, dummy): {wer_good:.4f}") # Expected 0.0
    wer_bad = calculate_wer(ref_ocr, hyp_ocr_bad)
    print(f"WER (bad, dummy): {wer_bad:.4f}") # Expected > 0.0 (3 substitutions / 3 words = 1.0 for this specific bad hyp)

    print("\nMetrics utility tests complete (placeholders).")
