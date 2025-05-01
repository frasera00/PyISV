import torch
import torch.nn as nn

class Classifier1D(nn.Module):
    """
    Fully self-contained 1D classifier using a CNN encoder and classification head.
    Expects input shape: [batch_size, 1, sequence_length]
    """
    def __init__(self, input_shape=(1, 200), embed_dim=128, num_classes=6, num_encoder_final_channels=16):
        super().__init__()
        
        # Encoder using 1D convolutions
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),
            
            nn.Conv1d(8, num_encoder_final_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),
        )

        # Automatically infer flat_dim
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.encoder(dummy_input)
            self.flat_dim = dummy_output.view(1, -1).shape[1]

        # Classification head
        self.fc1 = nn.Linear(self.flat_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        # Forward pass
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.relu(self.fc1(z))
        z = self.dropout(z)

        logits = self.fc2(z)
        return logits