# src/nlp/utils.py
# Utility functions for the NLP module

import re
# import nltk # For sentence tokenization, word tokenization if used
# from nltk.tokenize import sent_tokenize, word_tokenize
# import langdetect # For language detection, if used
# from transformers import AutoTokenizer # If using Hugging Face tokenizers for preprocessing

# Ensure NLTK data is available if using it (e.g., punkt for tokenization)
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     print("NLTK 'punkt' resource not found. Downloading...")
#     nltk.download('punkt', quiet=True)
# except LookupError: # Sometimes find raises LookupError instead of DownloadError
#     print("NLTK 'punkt' resource not found (LookupError). Downloading...")
#     nltk.download('punkt', quiet=True)


def preprocess_text_for_nlp(text, lower=False, remove_punctuation=False, language="en"):
    """
    Basic text preprocessing for NLP tasks.
    Args:
        text (str): Input text.
        lower (bool): Convert text to lowercase.
        remove_punctuation (bool): Remove common punctuation.
        language (str): Language code (e.g., "en", "hi") for language-specific rules if any.
                        (Currently not used in this placeholder).
    Returns:
        str: Preprocessed text.
    """
    print(f"Preprocessing text (placeholder in nlp.utils): '{text[:50]}...'")
    if not isinstance(text, str):
        print("Warning: Input text is not a string. Returning as is.")
        return text

    processed_text = text
    if lower:
        processed_text = processed_text.lower()
        print("  Applied lowercasing.")

    if remove_punctuation:
        # Basic punctuation removal, can be extended
        # For multilingual text, punctuation rules can be complex.
        # This is a very simple English-centric example.
        punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        processed_text = processed_text.translate(str.maketrans('', '', punctuation))
        print("  Applied punctuation removal (basic).")

    # Other common steps:
    # - Remove extra whitespace:
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    # - Normalize Unicode (see ocr.utils.normalize_text, could be used here too)
    # - Lemmatization or stemming (requires more advanced tools like spaCy, NLTK with WordNet)
    # - Stop word removal (language-dependent)

    print(f"Preprocessed text (placeholder): '{processed_text[:50]}...'")
    return processed_text


def detect_language(text, method="langdetect"):
    """
    Detects the language of a given text.
    Args:
        text (str): Input text.
        method (str): Library to use for language detection.
                      "langdetect" is a common choice.
    Returns:
        str: Detected language code (e.g., "en", "hi"), or "unknown".
    """
    print(f"Detecting language for text (placeholder in nlp.utils): '{text[:50]}...'")
    if not text or not isinstance(text, str) or len(text.strip()) < 10: # Too short to detect reliably
        print("  Text too short or invalid for language detection, returning 'unknown'.")
        return "unknown"

    if method == "langdetect":
        # try:
        #     detected_lang = langdetect.detect(text)
        #     print(f"  Detected language (langdetect): {detected_lang}")
        #     return detected_lang
        # except langdetect.lang_detect_exception.LangDetectException:
        #     print("  Langdetect could not detect language, returning 'unknown'.")
        #     return "unknown"
        # except Exception as e:
        #     print(f"  Error during langdetect: {e}. Returning 'unknown'.")
        #     return "unknown"
        # Placeholder:
        if "नमस्ते" in text or "भारत" in text: return "hi"
        if "Hello" in text or "India" in text: return "en"
        return "en" # Default placeholder
    else:
        print(f"Unsupported language detection method: {method}. Returning 'unknown'.")
        return "unknown"


def tokenize_text(text, method="nltk_word", language="english"):
    """
    Tokenizes text into words or subwords.
    Args:
        text (str): Input text.
        method (str): "nltk_word", "nltk_sent", "hf_tokenizer" (requires tokenizer path/name in future).
        language (str): Language for NLTK tokenizers (e.g., "english", "hindi").
    Returns:
        list: List of tokens (strings).
    """
    print(f"Tokenizing text with method '{method}' (placeholder in nlp.utils): '{text[:50]}...'")
    if not isinstance(text, str): return []

    # if method == "nltk_word":
    #     # Ensure NLTK's punkt is downloaded for word_tokenize for some languages
    #     try:
    #         return word_tokenize(text, language=language)
    #     except Exception as e:
    #         print(f"Error during NLTK word tokenization for language '{language}': {e}. Falling back to split().")
    #         return text.split()
    # elif method == "nltk_sent":
    #     try:
    #         return sent_tokenize(text, language=language)
    #     except Exception as e:
    #         print(f"Error during NLTK sentence tokenization for language '{language}': {e}. Falling back to splitlines().")
    #         return text.splitlines()
    # elif method == "hf_tokenizer":
    #     # This would require loading a specific Hugging Face tokenizer
    #     # tokenizer = AutoTokenizer.from_pretrained("your_tokenizer_name_or_path")
    #     # return tokenizer.tokenize(text)
    #     print("  Hugging Face tokenizer placeholder - returning word split.")
    #     return text.split()
    # else:
    #     print(f"  Unsupported tokenization method: {method}. Using simple split().")
    #     return text.split()

    # Placeholder:
    if method == "nltk_sent":
        return [s.strip() for s in text.split('.') if s.strip()] # Very basic sentence split
    return text.split() # Basic word split


if __name__ == '__main__':
    print("Testing NLP utility functions (placeholders)...")

    sample_text_en = "Hello World! This is a test sentence. Isn't it great? Let's process this. "
    sample_text_hi = "नमस्ते दुनिया! यह एक परीक्षण वाक्य है। क्या यह बढ़िया नहीं है? चलिये इसे प्रोसेस करते हैं।"

    # Test preprocess_text_for_nlp
    print("\n--- Testing preprocess_text_for_nlp ---")
    processed_en = preprocess_text_for_nlp(sample_text_en, lower=True, remove_punctuation=True)
    print(f"Original EN: {sample_text_en}")
    print(f"Processed EN (dummy): {processed_en}")

    processed_hi = preprocess_text_for_nlp(sample_text_hi, lower=False, remove_punctuation=False) # Typically don't lowercase Indic without thought
    print(f"Original HI: {sample_text_hi}")
    print(f"Processed HI (dummy): {processed_hi}")

    # Test detect_language
    print("\n--- Testing detect_language ---")
    lang_en = detect_language(sample_text_en)
    print(f"Detected language for English text (dummy): {lang_en}")
    assert lang_en == "en"

    lang_hi = detect_language(sample_text_hi)
    print(f"Detected language for Hindi text (dummy): {lang_hi}")
    assert lang_hi == "hi"

    lang_short = detect_language("Test.")
    print(f"Detected language for short text (dummy): {lang_short}")
    assert lang_short == "unknown"

    # Test tokenize_text
    print("\n--- Testing tokenize_text ---")
    tokens_en_word = tokenize_text(sample_text_en, method="nltk_word")
    print(f"English word tokens (dummy): {tokens_en_word[:10]}...")

    tokens_en_sent = tokenize_text(sample_text_en, method="nltk_sent")
    print(f"English sentence tokens (dummy): {tokens_en_sent}")

    tokens_hi_word = tokenize_text(sample_text_hi, method="nltk_word", language="hindi") # NLTK needs language for some scripts
    print(f"Hindi word tokens (dummy): {tokens_hi_word[:10]}...")

    print("\nNLP utility tests complete (placeholders).")
