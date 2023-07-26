import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Define Encoder Model (same as before)

# Define Decoder Model (same as before)

# Define HT-VAE Model (same as before)

# Load the syntax vectors (Assuming you have generated and saved them using paser.py)
syntax_vectors = []
for i in range(600):
    with open(f"{i}_syntax_vector.txt", "r") as f:
        syntax_vector = f.read().splitlines()
    syntax_vectors.append(syntax_vector)

# Preprocess the syntax vectors (convert to numpy array and pad if necessary)
max_seq_length = max(len(seq) for seq in syntax_vectors)
padded_syntax_vectors = [seq + ['PAD'] * (max_seq_length - len(seq)) for seq in syntax_vectors]
padded_syntax_vectors = np.array(padded_syntax_vectors)

# Create PyTorch DataLoader
batch_size = 32
syntax_tensor = torch.tensor(padded_syntax_vectors, dtype=torch.float32)
dataset = TensorDataset(syntax_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model hyperparameters (same as before)

# Build and train the HT-VAE model (same as before)

# Load the trained model
htvae = HT_VAE(input_shape, latent_dim)
htvae.load_state_dict(torch.load("trained_htvae_model.pt"))
htvae.eval()

# Generate samples using the trained decoder
def generate_samples(htvae, num_samples, max_seq_length, latent_dim):
    samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Initialize the input with Gaussian distribution and SOS identifier
            input_latent = torch.randn(1, latent_dim)
            sos_token = torch.tensor([[1] + [0] * (max_seq_length - 1)], dtype=torch.float32)
            sos_token = sos_token.unsqueeze(1)  # Add batch dimension

            # Generate sequence using the trained decoder
            generated_sequence = []
            decoder_input = torch.cat((input_latent, sos_token), dim=2)  # Concatenate latent input and SOS identifier

            for step in range(max_seq_length):
                decoder_output = htvae.decoder(decoder_input)
                generated_token = torch.argmax(decoder_output[0, 0, :])  # Get the most probable token index

                if generated_token == 0:  # EOS token
                    break

                generated_sequence.append(generated_token.item())
                decoder_input = torch.cat((input_latent, decoder_output[:, :, :step+2]), dim=2)  # Update decoder input

            samples.append(generated_sequence)

    return samples

# Generate 1000 samples
num_samples = 1000
generated_samples = generate_samples(htvae, num_samples, max_seq_length, latent_dim)

# Save the generated samples in the 'sequence' folder
if not os.path.exists("sequence"):
    os.mkdir("sequence")

for i, sample in enumerate(generated_samples):
    filename = os.path.join("sequence", f"{i}.txt")
    with open(filename, "w") as f:
        f.write(" ".join(str(token) for token in sample))

print("Generated samples saved in the 'sequence' folder.")