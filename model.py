"""
Transformer-based LLM Model with Autoregressive Generation
This module implements a GPT-style decoder-only transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask is True for positions to mask (prevent attention)
            scores = scores.masked_fill(mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations and split into heads
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Reshape mask for multi-head attention: [batch_size, seq_len, seq_len] -> [batch_size, num_heads, seq_len, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Apply attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.w_o(attn_output)
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
    def gelu(self, x):
        """GELU activation function - better for transformers than ReLU."""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        
    def forward(self, x):
        if self.activation == 'gelu':
            return self.linear2(self.dropout(self.gelu(self.linear1(x))))
        else:
            return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer decoder block with masked self-attention.
    
    Uses Pre-LN architecture (layer norm before attention/FF) which trains
    faster and more stably than Post-LN.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_pre_ln=True):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation='gelu')
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.use_pre_ln = use_pre_ln
        
    def forward(self, x, mask=None):
        if self.use_pre_ln:
            # Pre-LN: Layer norm BEFORE attention/FF (faster convergence)
            # Self-attention with residual connection
            attn_output = self.attention(self.norm1(x), mask)
            x = x + self.dropout1(attn_output)
            
            # Feed-forward with residual connection
            ff_output = self.feed_forward(self.norm2(x))
            x = x + self.dropout2(ff_output)
        else:
            # Post-LN: Layer norm AFTER attention/FF (original architecture)
            attn_output = self.attention(x, mask)
            x = self.norm1(x + self.dropout1(attn_output))
            
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Extend positional encoding if needed
            pe_extended = torch.zeros(1, seq_len, self.pe.size(2), device=x.device)
            pe_extended[:, :self.pe.size(1), :] = self.pe
            position = torch.arange(self.pe.size(1), seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.pe.size(2), 2, dtype=torch.float, device=x.device) * 
                               (-math.log(10000.0) / self.pe.size(2)))
            pe_extended[:, self.pe.size(1):, 0::2] = torch.sin(position * div_term)
            pe_extended[:, self.pe.size(1):, 1::2] = torch.cos(position * div_term)
            return x + pe_extended
        return x + self.pe[:, :seq_len]


class TransformerLLM(nn.Module):
    """
    GPT-style decoder-only Transformer Language Model.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Dimension of model embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        d_ff: Dimension of feed-forward network
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_seq_len=1024, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, use_pre_ln=True)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters with improved initialization for faster convergence."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize transformer blocks
        for block in self.transformer_blocks:
            # Attention weights
            nn.init.normal_(block.attention.w_q.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.attention.w_k.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.attention.w_v.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.attention.w_o.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
            
            # Feed-forward weights
            nn.init.normal_(block.feed_forward.linear1.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.feed_forward.linear2.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
            
            # Initialize biases to zero
            if block.attention.w_q.bias is not None:
                nn.init.zeros_(block.attention.w_q.bias)
            if block.attention.w_k.bias is not None:
                nn.init.zeros_(block.attention.w_k.bias)
            if block.attention.w_v.bias is not None:
                nn.init.zeros_(block.attention.w_v.bias)
            if block.attention.w_o.bias is not None:
                nn.init.zeros_(block.attention.w_o.bias)
            if block.feed_forward.linear1.bias is not None:
                nn.init.zeros_(block.feed_forward.linear1.bias)
            if block.feed_forward.linear2.bias is not None:
                nn.init.zeros_(block.feed_forward.linear2.bias)
        
        # Output layer - use smaller initialization for better stability
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)
    
    def generate_mask(self, size):
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input token indices [batch_size, seq_len]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len = x.size()
        
        # Generate causal mask if not provided
        if mask is None:
            mask = self.generate_mask(seq_len).to(x.device)
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Final layer norm and output projection
        x = self.layer_norm(x)
        logits = self.fc_out(x)
        
        return logits
    
    def generate(self, tokenizer, prompt="", max_length=100, temperature=1.0, 
                 top_k=50, top_p=0.9, device='cpu', filter_empty_tokens=True):
        """
        Autoregressive text generation.
        
        Args:
            tokenizer: Tokenizer instance
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            device: Device to run on
            filter_empty_tokens: If True, filter out tokens that decode to empty strings
        """
        self.eval()
        
        # Tokenize prompt
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long).to(device)
        
        generated = tokens.clone()
        
        # Identify tokens to filter out (empty/problematic tokens)
        tokens_to_filter = set()
        if filter_empty_tokens:
            # Filter out token 2 specifically (known to decode to empty string)
            tokens_to_filter.add(2)
            
            # Filter out pad token (0) and unk token (1) if they exist
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                tokens_to_filter.add(tokenizer.pad_token_id)
            if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
                tokens_to_filter.add(tokenizer.unk_token_id)
            
            # Check a sample of tokens to find other empty ones
            # (checking first 100 tokens should catch most special tokens)
            vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else 1000
            for token_id in range(min(100, vocab_size)):
                if token_id in tokens_to_filter:
                    continue
                try:
                    decoded = tokenizer.decode([token_id])
                    if decoded == '' or (decoded.strip() == '' and token_id not in [0, 1, 2, 3]):
                        tokens_to_filter.add(token_id)
                except:
                    pass
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(generated)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Filter out empty/problematic tokens BEFORE other filtering
                if filter_empty_tokens and tokens_to_filter:
                    for token_id in tokens_to_filter:
                        if token_id < len(next_token_logits):
                            next_token_logits[token_id] = float('-inf')
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Check if all tokens are filtered out (shouldn't happen, but safety check)
                if torch.all(next_token_logits == float('-inf')):
                    # If all filtered, reset and only filter empty tokens
                    next_token_logits = logits[0, -1, :] / temperature
                    if filter_empty_tokens and tokens_to_filter:
                        for token_id in tokens_to_filter:
                            if token_id < len(next_token_logits):
                                next_token_logits[token_id] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop if end token is generated (if tokenizer has one)
                # Note: We allow EOS token to stop generation, but filter it during sampling
                if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode generated tokens
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        return generated_text

