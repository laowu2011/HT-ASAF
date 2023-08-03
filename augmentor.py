import os
import torch
import torch.nn as nn
import sys
sys.path.append("..//")
from model.HTVAE import HTVAE
from model.HTVAE import tokenize_input
from model.HTVAE import build_vocab
from model.HTVAE import tokens_to_indices

# ... [Only keep the required functions and class definitions]

def generate_samples(vae, num_samples, seq_length, vocab):
    vae.eval()  # Set the VAE to evaluation mode
    reverse_vocab = {v: k for k, v in vocab.items()}  # Create a reverse vocabulary

    generated_samples = []

    for _ in range(num_samples):
        z = torch.randn((1, seq_length, vae.mean_layer.out_features)).to(device)
        with torch.no_grad():
            dec_output, _ = vae.decoder_rnn(z)
            x_hat = vae.decoder_output(dec_output)

        # Get the most probable tokens
        generated_seq = torch.argmax(x_hat.squeeze(), dim=-1).cpu().numpy()

        # Convert indices back to tokens
        generated_tokens = [reverse_vocab[idx] for idx in generated_seq if idx != 0]  # Ignore the PAD token
        generated_samples.append(" ".join(generated_tokens))

    return generated_samples

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Build vocabulary from all files in the original data directory
    data_dir = '../data/AES-T_Sequence'
    all_tokens = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                tokens = tokenize_input(content)
                all_tokens.extend(tokens)

    vocab = build_vocab(all_tokens)

    # Initialize the VAE model
    vae = HTVAE(len(vocab))

    # Load the trained model
    vae.load_state_dict(torch.load('../model/trained_ht_vae.pt'))
    vae = vae.to(device)

    # Generate samples
    generated_samples = generate_samples(vae, 1000, 354, vocab)
    
    # Save generated samples
    save_dir = '../data/AES-T_Aug_Sequence'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, sample in enumerate(generated_samples):
        with open(os.path.join(save_dir, f'generated_sample_{i}.txt'), 'w', encoding='utf-8') as f:
            f.write(sample)
