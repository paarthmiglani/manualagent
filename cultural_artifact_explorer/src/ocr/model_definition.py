# src/ocr/model_definition.py
# Defines the neural network architecture for the OCR model.

import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    """
    Custom CRNN (Convolutional Recurrent Neural Network) model for OCR.
    This architecture consists of three main parts:
    1. CNN (Convolutional Neural Network) for visual feature extraction.
    2. RNN (Recurrent Neural Network) for sequence modeling from the extracted features.
    3. Transcription layer (fully connected) for predicting characters at each time step.
    """
    def __init__(self, img_channels, num_classes, rnn_hidden_size=256, rnn_num_layers=2, dropout=0.5):
        """
        Initializes the CRNN model.
        Args:
            img_channels (int): Number of channels in the input image (1 for grayscale, 3 for color).
            num_classes (int): Number of classes to predict (size of character set + 1 for blank token).
            rnn_hidden_size (int): Number of hidden units in the RNN layers.
            rnn_num_layers (int): Number of layers in the RNN (e.g., stacked LSTM).
            dropout (float): Dropout probability for RNN layers.
        """
        super(CRNN, self).__init__()

        # --- 1. Convolutional Layers (CNN Backbone) ---
        # The goal is to extract features from the image. The architecture below is a common pattern.
        # Input image size is assumed to be (Batch, Channels, Height, Width), e.g., (B, 1, 32, W).
        # The CNN should reduce the height dimension significantly while transforming the width into a sequence length.

        self.cnn_backbone = nn.Sequential(
            # Input: (B, img_channels, 32, W) -> Output: (B, 64, 16, W/2)
            nn.Conv2d(img_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Input: (B, 64, 16, W/2) -> Output: (B, 128, 8, W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Input: (B, 128, 8, W/4) -> Output: (B, 256, 4, W/8)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), # Add batch norm for stability
            nn.ReLU(inplace=True),

            # Additional conv layer
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)), # (H,W) pool, stride, pad

            # Input: (B, 256, 2, W/4) -> Output: (B, 512, 1, W/4)
            # A conv with non-square kernel can reduce height to 1
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        )
        # After the last MaxPool2d, the feature map should have a height of 1.
        # Let's verify the output shape with a dummy forward pass in the comments.
        # If input H=32:
        # 32 -> 16 (pool1) -> 8 (pool2) -> 4 (pool3 H=2,S=2) -> 2 (pool4 H=2,S=2) -> This is wrong.
        # Let's adjust the pooling/conv layers to ensure H becomes 1.

        # Corrected CNN to ensure H -> 1 for input H=32
        self.cnn_backbone_corrected = nn.Sequential(
            # Input: (B, C, 32, W)
            nn.Conv2d(img_channels, 32, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # -> (B, 32, 16, W/2)
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # -> (B, 64, 8, W/4)
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # -> (B, 128, 4, W/4 + 1)
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # -> (B, 256, 2, W/4 + 2)
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1))  # -> (B, 512, 1, W/4 + 3)
        )
        # The output width will be the sequence length for the RNN.
        # The feature map of shape (B, 512, 1, SeqLen) needs to be reshaped.


        # --- 2. Recurrent Layers (RNN) ---
        # The output of the CNN is a feature map. We treat the width dimension as a sequence.
        # The features at each "time step" (column) are fed into the RNN.
        # We need to determine the input size for the RNN based on the CNN output.
        # After the CNN, we have (Batch, Channels, 1, Width). We squeeze H and treat W as sequence.
        # RNN Input: (SeqLen, Batch, Features)
        # Features = Channels from CNN = 512 in this case.

        self.rnn = nn.LSTM(
            input_size=512, # From CNN output channels
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True, # Bidirectional LSTM is standard for OCR
            batch_first=False, # We will permute the data to be (SeqLen, Batch, Feat)
            dropout=dropout if rnn_num_layers > 1 else 0
        )

        # --- 3. Transcription Layer (Fully Connected) ---
        # Maps the RNN output to character predictions.
        # RNN output is (SeqLen, Batch, HiddenSize * 2) because it's bidirectional.
        self.transcription_layer = nn.Linear(
            rnn_hidden_size * 2, # *2 for bidirectional
            num_classes # Number of characters + blank
        )

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).
        Returns:
            torch.Tensor: Log probabilities of character predictions.
                          Shape: (SeqLen, Batch, NumClasses), ready for CTC loss.
        """
        # 1. Pass through CNN
        features = self.cnn_backbone_corrected(x)
        # features shape: (B, C, H, W) -> (B, 512, 1, SeqLen)

        # 2. Prepare for RNN (Map-to-Sequence)
        # Squeeze the height dimension and permute to fit RNN input format (SeqLen, Batch, Features)
        b, c, h, w = features.size()
        assert h == 1, "The height of CNN output must be 1"
        features = features.squeeze(2) # -> (B, 512, SeqLen)
        features = features.permute(2, 0, 1) # -> (SeqLen, B, 512)

        # 3. Pass through RNN
        rnn_output, _ = self.rnn(features)
        # rnn_output shape: (SeqLen, B, HiddenSize * 2)

        # 4. Pass through Transcription Layer
        output = self.transcription_layer(rnn_output)
        # output shape: (SeqLen, B, NumClasses)

        # 5. Apply Log Softmax for CTC Loss
        # The CTC loss function in PyTorch expects log probabilities.
        # It's efficient to do log_softmax here.
        log_probs = F.log_softmax(output, dim=2)

        return log_probs


if __name__ == '__main__':
    print("--- Testing CRNN Model Definition ---")

    # Automatic device selection (MPS for Apple Silicon, CUDA, else CPU)
    import torch
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # --- Model Parameters (Example) ---
    IMG_CHANNELS = 1 # Grayscale
    NUM_CLASSES = 80 # Example: 79 characters + 1 blank token
    RNN_HIDDEN = 256
    RNN_LAYERS = 2

    # --- Create Model Instance ---
    model = CRNN(
        img_channels=IMG_CHANNELS,
        num_classes=NUM_CLASSES,
        rnn_hidden_size=RNN_HIDDEN,
        rnn_num_layers=RNN_LAYERS
    ).to(device)
    print("CRNN model instance created successfully.")
    print(model)

    # --- Test Forward Pass with Dummy Data ---
    print("\n--- Testing forward pass ---")
    BATCH_SIZE = 4
    IMG_HEIGHT = 32
    IMG_WIDTH = 150 # Example width

    # Create a dummy input tensor
    dummy_input = torch.randn(BATCH_SIZE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, device=device)
    print(f"Dummy input tensor shape: {dummy_input.shape}")

    try:
        # Perform a forward pass
        output_log_probs = model(dummy_input)
        print(f"Model output (log_probs) shape: {output_log_probs.shape}")

        # Check output shape
        cnn_output = model.cnn_backbone_corrected(dummy_input)
        expected_seq_len = cnn_output.size(3)
        print(f"CNN output feature map shape: {cnn_output.shape}")
        print(f"Inferred sequence length for RNN: {expected_seq_len}")

        expected_shape = (expected_seq_len, BATCH_SIZE, NUM_CLASSES)
        if output_log_probs.shape == expected_shape:
            print(f"Forward pass test PASSED. Output shape {output_log_probs.shape} matches expected {expected_shape}.")
        else:
            print(f"Forward pass test FAILED. Output shape {output_log_probs.shape} does not match expected {expected_shape}.")

    except Exception as e:
        print(f"An error occurred during the forward pass test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Model Definition Script Finished ---")
