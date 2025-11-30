"""
Improved Transformer Architecture (Llama-style)
Features: RMSNorm, RoPE, SwiGLU, Flash Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply Rotary Positional Embeddings (RoPE) to query and key tensors.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Bias=False is better for modern LLMs
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = dropout

    def forward(self, x, freqs_cis, mask=None):
        batch_size, seq_len, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for multi-head attention
        xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Transpose for attention: (bs, heads, seq, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Flash Attention (Scaled Dot Product Attention)
        # Handles masking internally if provided, or we can use is_causal=True for causal masking
        if mask is not None:
             # Convert boolean mask to attention bias if needed, but F.scaled_dot_product_attention 
             # handles causal masking efficiently with is_causal=True
             pass
             
        # Using PyTorch 2.0+ scaled_dot_product_attention
        # If we are doing causal training, we can often just use is_causal=True
        # However, for flexibility with custom masks (if any), we'll pass is_causal=True 
        # which implies a lower triangular mask.
        output = F.scaled_dot_product_attention(
            xq, xk, xv, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    """
    SwiGLU FeedForward module
    """
    def __init__(self, d_model, d_ff, multiple_of=256, dropout=0.0):
        super().__init__()
        hidden_dim = int(2 * d_ff / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: w2(F.silu(w1(x)) * w3(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = Attention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        
        # Pre-normalization with RMSNorm
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x, freqs_cis, mask=None):
        # Residual connections
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class ImprovedTransformerLLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Precompute RoPE frequencies
        self.head_dim = d_model // num_heads
        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Move freqs_cis to correct device if needed
        if self.freqs_cis.device != x.device:
            self.freqs_cis = self.freqs_cis.to(x.device)
            
        freqs_cis = self.freqs_cis[:seq_len]
        
        h = self.token_embedding(x)
        h = self.dropout(h)
        
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
            
        h = self.norm(h)
        return self.output(h)

    @torch.no_grad()
    def generate(self, tokenizer, prompt="", max_length=100, temperature=1.0, 
                 top_k=50, top_p=0.9, device='cpu', filter_empty_tokens=True):
        self.eval()
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long).to(device)
        
        for _ in range(max_length):
            # Crop context if too long
            context = tokens[:, -self.max_seq_len:]
            
            logits = self.forward(context)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Filter
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[[-1]]] = float('-inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
                break
                
        return tokenizer.decode(tokens[0].cpu().tolist())
