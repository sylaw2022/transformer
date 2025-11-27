"""
Simple tokenizer for text processing.
Supports character-level and word-level tokenization.
"""

from collections import Counter
from typing import List, Dict, Optional


class SimpleTokenizer:
    """
    Simple tokenizer that supports both character-level and word-level tokenization.
    """
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None, 
                 vocab_size: Optional[int] = None,
                 mode='char'):
        """
        Initialize tokenizer.
        
        Args:
            vocab: Optional vocabulary dictionary {token: id}
            vocab_size: Optional vocabulary size (for character-level)
            mode: 'char' for character-level or 'word' for word-level
        """
        self.mode = mode
        self.vocab = vocab or {}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'
        self.bos_token = '<BOS>'
        
        # Special token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self.bos_token_id = 3
        
        # Initialize special tokens if vocab is empty
        if not self.vocab:
            self.vocab = {
                self.pad_token: self.pad_token_id,
                self.unk_token: self.unk_token_id,
                self.eos_token: self.eos_token_id,
                self.bos_token: self.bos_token_id,
            }
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts: List[str], vocab_size: Optional[int] = None):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            vocab_size: Maximum vocabulary size (None for all tokens)
        """
        if self.mode == 'char':
            # Character-level: collect all unique characters
            all_chars = set()
            for text in texts:
                all_chars.update(text)
            
            # Sort characters for consistency
            sorted_chars = sorted(all_chars)
            
            # Build vocab (reserve first 4 for special tokens)
            self.vocab = {
                self.pad_token: self.pad_token_id,
                self.unk_token: self.unk_token_id,
                self.eos_token: self.eos_token_id,
                self.bos_token: self.bos_token_id,
            }
            
            for idx, char in enumerate(sorted_chars):
                self.vocab[char] = idx + 4
            
            if vocab_size:
                # Limit vocab size
                sorted_items = sorted(self.vocab.items(), key=lambda x: x[1])
                self.vocab = dict(sorted_items[:vocab_size])
                self.vocab[self.pad_token] = self.pad_token_id
                self.vocab[self.unk_token] = self.unk_token_id
                self.vocab[self.eos_token] = self.eos_token_id
                self.vocab[self.bos_token] = self.bos_token_id
                
        else:  # word-level
            # Count word frequencies
            import re
            word_counts = Counter()
            for text in texts:
                # Better word tokenization: split on whitespace and punctuation
                # Keep punctuation as separate tokens for better handling
                words = re.findall(r'\b\w+\b|[.,!?;:()"\'-]', text.lower())
                word_counts.update(words)
            
            # Build vocab with most frequent words
            self.vocab = {
                self.pad_token: self.pad_token_id,
                self.unk_token: self.unk_token_id,
                self.eos_token: self.eos_token_id,
                self.bos_token: self.bos_token_id,
            }
            
            most_common = word_counts.most_common(vocab_size - 4 if vocab_size else None)
            for idx, (word, _) in enumerate(most_common):
                self.vocab[word] = idx + 4
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, add_bos=False, add_eos=False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
        """
        tokens = []
        
        if add_bos:
            tokens.append(self.bos_token_id)
        
        if self.mode == 'char':
            for char in text:
                tokens.append(self.vocab.get(char, self.unk_token_id))
        else:  # word-level
            import re
            # Better word tokenization: split on whitespace and punctuation
            words = re.findall(r'\b\w+\b|[.,!?;:()"\'-]', text.lower())
            for word in words:
                tokens.append(self.vocab.get(word, self.unk_token_id))
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
        """
        tokens = []
        for token_id in token_ids:
            if token_id == self.pad_token_id:
                continue
            if token_id == self.bos_token_id:
                continue
            if token_id == self.eos_token_id:
                break
            token = self.reverse_vocab.get(token_id, self.unk_token)
            tokens.append(token)
        
        if self.mode == 'char':
            return ''.join(tokens)
        else:  # word-level
            # Join words with spaces, but handle punctuation properly
            result = []
            for i, token in enumerate(tokens):
                if token in '.,!?;:()"\'-':
                    # Punctuation attaches to previous word (no space before)
                    result.append(token)
                    # Add space after punctuation if next token is a word
                    if i < len(tokens) - 1 and tokens[i + 1] not in '.,!?;:()"\'-':
                        result.append(' ')
                else:
                    # Add space before word if not first token
                    if result and result[-1] not in [' ', '']:
                        result.append(' ' + token)
                    else:
                        result.append(token)
            return ''.join(result).strip()
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.vocab)
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'mode': self.mode,
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'eos_token': self.eos_token,
                'bos_token': self.bos_token,
            }, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab=data['vocab'], mode=data['mode'])
        tokenizer.pad_token = data['pad_token']
        tokenizer.unk_token = data['unk_token']
        tokenizer.eos_token = data['eos_token']
        tokenizer.bos_token = data['bos_token']
        
        return tokenizer


