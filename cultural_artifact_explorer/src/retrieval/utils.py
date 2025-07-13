# src/retrieval/utils.py
# Utility functions for the Retrieval module

import numpy as np
# from PIL import Image # For image loading
# import torch # For tensor conversions if using PyTorch

def load_image_for_retrieval(image_path_or_data, preprocess_config=None, device='cpu'):
    """
    Loads and preprocesses an image for retrieval model input.
    Args:
        image_path_or_data (str, np.ndarray, PIL.Image): Path to image or loaded image.
        preprocess_config (dict, optional): Configuration for preprocessing, e.g.,
            { 'target_size': [224, 224],
              'mean': [0.485, 0.456, 0.406],
              'std': [0.229, 0.224, 0.225] }
        device (str): Device to move tensor to ('cpu', 'cuda:0').
    Returns:
        torch.Tensor: Preprocessed image as a tensor (Batch=1, C, H, W).
    """
    print(f"Loading and preprocessing image for retrieval (placeholder in retrieval.utils)...")
    # if isinstance(image_path_or_data, str):
    #     image = Image.open(image_path_or_data).convert('RGB')
    # elif isinstance(image_path_or_data, np.ndarray):
    #     image = Image.fromarray(image_path_or_data).convert('RGB')
    # elif isinstance(image_path_or_data, Image.Image):
    #     image = image_path_or_data.convert('RGB')
    # else:
    #     raise TypeError("Unsupported image_path_or_data type.")

    # if preprocess_config:
    #     target_size = tuple(preprocess_config.get('target_size', (224, 224))) # H, W -> W, H for PIL resize
    #     image = image.resize((target_size[1], target_size[0]))

    #     img_np = np.array(image).astype(np.float32) / 255.0 # To [0,1]

    #     mean = np.array(preprocess_config.get('mean', [0.485, 0.456, 0.406]))
    #     std = np.array(preprocess_config.get('std', [0.229, 0.224, 0.225]))
    #     img_np = (img_np - mean) / std

    #     # HWC to CHW
    #     img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1)))
    # else: # Minimal processing if no config
    #     img_np = np.array(image.resize((224,224))).astype(np.float32) / 255.0
    #     img_tensor = torch.from_numpy(img_np.transpose((2,0,1)))

    # return img_tensor.unsqueeze(0).to(device) # Add batch dimension

    # Placeholder: return a dummy tensor of expected shape
    c = 3
    h = preprocess_config.get('target_size', [224,224])[0] if preprocess_config else 224
    w = preprocess_config.get('target_size', [224,224])[1] if preprocess_config else 224
    # return torch.randn(1, c, h, w).to(device)
    return np.random.randn(1, c, h, w).astype(np.float32) # Numpy placeholder


def preprocess_text_for_retrieval(text, tokenizer, preprocess_config=None, device='cpu'):
    """
    Tokenizes and preprocesses text for retrieval model input.
    Args:
        text (str): Input text.
        tokenizer: Pre-initialized tokenizer (e.g., Hugging Face AutoTokenizer).
        preprocess_config (dict, optional): Configuration, e.g.,
            { 'max_length': 77, 'truncation': True, 'padding': 'max_length' }
        device (str): Device to move tensors to.
    Returns:
        dict: Dictionary of tokenized inputs (e.g., {'input_ids': ..., 'attention_mask': ...}).
    """
    print(f"Preprocessing text for retrieval (placeholder in retrieval.utils): '{text[:50]}...'")
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for text preprocessing.")

    # default_config = {'max_length': 77, 'truncation': True, 'padding': 'max_length'}
    # conf = preprocess_config if preprocess_config else default_config

    # tokenized_inputs = tokenizer(
    #     text,
    #     return_tensors="pt",
    #     max_length=conf.get('max_length'),
    #     truncation=conf.get('truncation'),
    #     padding=conf.get('padding')
    # )
    # return {k: v.to(device) for k, v in tokenized_inputs.items()}

    # Placeholder:
    max_len = preprocess_config.get('max_length', 77) if preprocess_config else 77
    dummy_input_ids = np.random.randint(0, 30000, size=(1, max_len)) # Batch=1, SeqLen=max_len
    dummy_attention_mask = np.ones((1, max_len))
    # return {
    #     'input_ids': torch.tensor(dummy_input_ids).to(device),
    #     'attention_mask': torch.tensor(dummy_attention_mask).to(device)
    # }
    return { # Numpy placeholder
        'input_ids': dummy_input_ids,
        'attention_mask': dummy_attention_mask
    }


def normalize_embedding(embedding_vector):
    """Normalizes a 1D embedding vector to unit length (L2 normalization)."""
    if not isinstance(embedding_vector, np.ndarray):
        embedding_vector = np.array(embedding_vector)

    norm = np.linalg.norm(embedding_vector)
    if norm == 0:
        return embedding_vector # Avoid division by zero
    return embedding_vector / norm


if __name__ == '__main__':
    print("Testing Retrieval utility functions (placeholders)...")

    # --- Test load_image_for_retrieval ---
    print("\n--- Testing load_image_for_retrieval ---")
    # Dummy image data (e.g., path or np array)
    # For path, you'd need a dummy image file. Let's use np array.
    dummy_img_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img_preprocess_conf = { 'target_size': [224, 224],
                            'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225] }

    try:
        # preprocessed_tensor = load_image_for_retrieval(dummy_img_np, img_preprocess_conf)
        # print(f"Loaded and preprocessed image tensor shape (dummy): {preprocessed_tensor.shape}")
        # assert list(preprocessed_tensor.shape) == [1, 3, 224, 224]
        # Placeholder version:
        preprocessed_np = load_image_for_retrieval(dummy_img_np, img_preprocess_conf)
        print(f"Loaded and preprocessed image numpy shape (dummy): {preprocessed_np.shape}")
        assert list(preprocessed_np.shape) == [1, 3, 224, 224]

    except Exception as e:
        print(f"Error in load_image_for_retrieval test: {e} (This might be due to missing PIL/Torch for real run)")


    # --- Test preprocess_text_for_retrieval ---
    print("\n--- Testing preprocess_text_for_retrieval ---")
    # Dummy tokenizer (mock object)
    class DummyTokenizer:
        def __init__(self, vocab_size=30000):
            self.vocab_size = vocab_size
        def __call__(self, text, return_tensors="pt", max_length=77, truncation=True, padding='max_length'):
            print(f"DummyTokenizer called for: '{text}', max_len={max_length}")
            # Simulate tokenization
            # input_ids = torch.randint(0, self.vocab_size, (1, max_length))
            # attention_mask = torch.ones((1, max_length))
            # return {'input_ids': input_ids, 'attention_mask': attention_mask}
            # Numpy placeholder:
            input_ids = np.random.randint(0, self.vocab_size, (1, max_length))
            attention_mask = np.ones((1, max_length))
            return {'input_ids': input_ids, 'attention_mask': attention_mask}


    dummy_tokenizer = DummyTokenizer()
    sample_text = "An ancient bronze statue of a deity."
    text_preprocess_conf = { 'max_length': 77, 'truncation': True, 'padding': 'max_length' }

    try:
        tokenized_output = preprocess_text_for_retrieval(sample_text, dummy_tokenizer, text_preprocess_conf)
        print(f"Tokenized text output (dummy):")
        print(f"  input_ids shape: {tokenized_output['input_ids'].shape}")
        # assert list(tokenized_output['input_ids'].shape) == [1, 77]
        # assert list(tokenized_output['attention_mask'].shape) == [1, 77]
    except Exception as e:
        print(f"Error in preprocess_text_for_retrieval test: {e} (This might be due to missing Torch for real run)")


    # --- Test normalize_embedding ---
    print("\n--- Testing normalize_embedding ---")
    test_vector = np.array([1.0, 2.0, 3.0, 4.0])
    normalized_vector = normalize_embedding(test_vector)
    norm_of_normalized = np.linalg.norm(normalized_vector)
    print(f"Original vector: {test_vector}")
    print(f"Normalized vector: {normalized_vector}")
    print(f"Norm of normalized vector: {norm_of_normalized}")
    assert np.isclose(norm_of_normalized, 1.0), "Normalized vector norm is not 1.0"

    zero_vector = np.array([0.0, 0.0, 0.0])
    normalized_zero = normalize_embedding(zero_vector)
    print(f"Normalized zero vector: {normalized_zero}")
    assert np.all(normalized_zero == 0.0), "Normalized zero vector is not all zeros"

    print("\nRetrieval utility tests complete (placeholders).")
