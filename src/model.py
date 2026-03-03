import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_hidden)
        self.fc2 = nn.Linear(cfg.d_hidden, cfg.d_model)

    def forward(self, x: Int[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def get_M(n):
    ones = torch.ones((n, n))
   
    mask = torch.triu(ones, diagonal=1)
    
    return mask.masked_fill(mask == 1, float('-inf'))
        
class AttentionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_qk = nn.Linear(cfg.d_model, cfg.d_model)
        self.W_ov = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: Int[torch.Tensor, "n_context d_model"]) -> Float[torch.Tensor, "n_context d_model"]:
        X = x
        n_context: int = X.shape[0]
        M = get_M(n_context)
        # print(f"{X.shape = }")
        # print(f"{M.shape = }")
        # print(f"{self.W_qk.shape = }")
        
        out = torch.softmax(X @ self.W_qk.weight.T @ X.T + M, dim=-1) @ X @ self.W_ov.weight.T
        return out

class Transformer_Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = MLP(cfg)
        self.attn_head = AttentionHead(cfg)
        
    def forward(self, x):
        x = x + self.attn_head(x) + self.mlp(x)
        return x

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # embed
        self.embedding = nn.Embedding(cfg.d_vocab, cfg.d_model)

        # pos embedding table
        self.pos_embedding = nn.Embedding(cfg.n_ctx, cfg.d_model)

        # list of blocks
        self.blocks = nn.ModuleList([Transformer_Block(cfg) for _ in range(cfg.n_layers)])

        #unembed
        self.unembedding = nn.Linear(cfg.d_model, cfg.d_vocab)

    def generate(self, prompt: str, max_iterations: int, tokenizer, cfg):
        #takes in text and tokenizes it
        input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
        self.eval()
        #run forward pass max_iterations times
        for i in range(max_iterations):
            idx_cond = input_ids[-cfg.n_ctx:]
            with torch.no_grad():
                logits = self.forward(idx_cond)

            last_logits = logits[-1, :]
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

            # probs = F.softmax(last_logits, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=0)
        #return generated text
        return tokenizer.decode(input_ids.tolist())
    
    def forward(self, x):
        
        # current num of tokens
        T = x.size(-1)

        # standard token embedding
        token_embeddings = self.embedding(x)

        # positional embedding
        positions = torch.arange(0, T)
        pos_embeddings = self.pos_embedding(positions)

        x = token_embeddings + pos_embeddings

        # apply each transformer block
        for block in self.blocks:
            x = block(x)

        # unembed
        x = self.unembedding(x)

        return x