import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Define Encoder Model
class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (input_shape[0]//4) * (input_shape[1]//4), 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, latent_dim)
        self.fc3 = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        z_mean = self.fc2(x)
        z_log_var = self.fc3(x)
        return z_mean, z_log_var

# Define Decoder Model
class Decoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, input_shape[0]*input_shape[1])
        self.relu2 = nn.ReLU()
        self.reshape = nn.Reshape(1, input_shape[0], input_shape[1])
        self.conv_transpose1 = nn.ConvTranspose2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.relu3 = nn.ReLU()
        self.conv_transpose2 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        self.relu4 = nn.ReLU()
        self.conv_transpose3 = nn.ConvTranspose2d(16, 1, kernel_size=(3, 3), padding=1)

    def forward(self, z):
        x = self.fc1(z)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.reshape(x)
        x = self.conv_transpose1(x)
        x = self.relu3(x)
        x = self.conv_transpose2(x)
        x = self.relu4(x)
        x = self.conv_transpose3(x)
        return x

# Define HT-VAE Model
class HT_VAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(HT_VAE, self).__init__()
        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(input_shape, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decoder(z)

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

# Define model hyperparameters
input_shape = (max_seq_length,)  # Replace this with the actual input shape of your syntax vectors
latent_dim = 32  # Replace this with the desired latent dimension

# Build and train the HT-VAE model
htvae = HT_VAE(input_shape, latent_dim)
optimizer = optim.Adam(htvae.parameters(), lr=0.001)

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

epochs = 900
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = batch[0]  # Remove the extra batch dimension added by DataLoader
        recon_batch = htvae(batch)
        loss = loss_function(recon_batch, batch, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

# Save the trained model if needed
# torch.save(htvae.state_dict(), "trained_htvae_model.pt")