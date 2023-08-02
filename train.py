import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Tokenize the input data
def tokenize_input(data):
    return [token for token in data.split() if token not in ['(', 'at', ')']]

# Build vocabulary from the tokens
def build_vocab(tokens):
    vocab = {'<PAD>': 0}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

# Convert tokens to indices based on the vocabulary
def tokens_to_indices(tokens, vocab):
    return [vocab.get(token, 0) for token in tokens]  # Use 0 as the default index for unknown tokens

class TextDataset(Dataset):
    def __init__(self, directory, vocab):
        self.data = []
        self.vocab = vocab
        
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):  # Assuming the text files have .txt extension
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    tokens = tokenize_input(content)
                    indexed_data = tokens_to_indices(tokens, self.vocab)
                    self.data.append(indexed_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # make sure all the returned sequences are 354
        sequence = self.data[idx]
        while len(sequence) < 354:
            sequence.append(0)  # Use the PAD index in vocab to populate
        return torch.LongTensor(sequence)

# Define the VAE model
class HTVAE(nn.Module):
    def __init__(self, vocab_size, latent_dim=20, embed_dim=128, hidden_dim=256):
        super(HTVAE, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM-encoder
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.var_layer = nn.Linear(hidden_dim, latent_dim)
        
        # LSTM-decoder
        self.decoder_rnn = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.decoder_output = nn.Linear(hidden_dim, vocab_size)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        x_embed = self.embedding(x)
        _, (h_n, _) = self.encoder_rnn(x_embed)
        
        h_n = h_n.squeeze(0)
        
        mean = self.mean_layer(h_n)
        log_var = self.var_layer(h_n)
        z = self.reparameterize(mean, log_var)
        
        # Here we copy z so that it matches the length of the input sequence
        z_repeated = z.unsqueeze(1).repeat(1, x.size(1), 1)
        
        dec_output, _ = self.decoder_rnn(z_repeated)
        x_hat = self.decoder_output(dec_output)
        
        return x_hat, mean, log_var

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Calculate VAE loss
def vae_loss(x, x_hat, mean, log_var):
    # nn.CrossEntropyLoss expects inputs to be probabilities
    # and targets to be class indices
    x = x.long()  # Make sure x is LongTensor
    x_hat = x_hat.view(-1, x_hat.size(-1))  # Reshape x_hat to 2D tensor
    x = x.view(-1)  # Reshape x to 1D tensor
    
    reconstruction_loss = nn.CrossEntropyLoss(reduction='sum')(x_hat, x)
    
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
    return reconstruction_loss + kl_divergence

def train_vae(vae, dataloader, num_epochs=1500, learning_rate=0.001):
    vae = vae.to(device)  # Move the model to the appropriate device
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, mean, log_var = vae(batch)
            loss = vae_loss(batch.float(), x_hat, mean, log_var)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(vae.state_dict(), 'trained_ht_vae.pt')

if __name__ == '__main__':
    data_dir = '../data/AES-T_Sequence'
    
    # Build vocabulary from all files
    all_tokens = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                tokens = tokenize_input(content)
                all_tokens.extend(tokens)

    vocab = build_vocab(all_tokens)
    
    # Load data
    dataset = TextDataset(data_dir, vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True))

    # Initialize the VAE model
    vae = HTVAE(len(vocab))

    # Train the VAE
    train_vae(vae, dataloader)
