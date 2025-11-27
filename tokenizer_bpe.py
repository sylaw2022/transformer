"""
BPE (Byte Pair Encoding) tokenizer using HuggingFace tokenizers library.
This provides subword tokenization which is more efficient than word-level.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
import json
import os
from typing import List, Optional, Dict


class BPETokenizer:
    """
    BPE tokenizer wrapper that matches SimpleTokenizer interface.
    """
    
    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        """
        Initialize BPE tokenizer.
        
        Args:
            tokenizer: Optional pre-trained HuggingFace Tokenizer instance
        """
        if tokenizer is None:
            self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
            self.tokenizer.pre_tokenizer = Whitespace()
        else:
            self.tokenizer = tokenizer
        
        # Special tokens (matching SimpleTokenizer interface)
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'
        self.bos_token = '<BOS>'
        
        # Special token IDs (will be set after training)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self.bos_token_id = 3
    
    def train(self, texts: List[str], vocab_size: int = 30000):
        """
        Train BPE tokenizer on texts.
        
        Args:
            texts: List of text strings
            vocab_size: Target vocabulary size
        """
        # Create temporary file with training data
        temp_file = 'temp_bpe_training.txt'
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # Train tokenizer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<EOS>", "<BOS>"],
            min_frequency=2
        )
        
        self.tokenizer.train(files=[temp_file], trainer=trainer)
        
        # Set up post-processor
        self.tokenizer.post_processor = BertProcessing(
            ("<EOS>", self.tokenizer.token_to_id("<EOS>")),
            ("<BOS>", self.tokenizer.token_to_id("<BOS>")),
        )
        
        # Update special token IDs
        self.pad_token_id = self.tokenizer.token_to_id("<PAD>")
        self.unk_token_id = self.tokenizer.token_to_id("<UNK>")
        self.eos_token_id = self.tokenizer.token_to_id("<EOS>")
        self.bos_token_id = self.tokenizer.token_to_id("<BOS>")
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    def encode(self, text: str, add_bos=False, add_eos=False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
        """
        # BPE tokenizer handles special tokens via post-processor
        # We'll manually add BOS/EOS if needed
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        
        if add_bos:
            token_ids = [self.bos_token_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_token_id]
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
        """
        # Filter out special tokens for decoding
        filtered_ids = [
            tid for tid in token_ids
            if tid not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]
        ]
        
        decoded = self.tokenizer.decode(filtered_ids)
        return decoded
    
    def __len__(self):
        """Return vocabulary size."""
        return self.tokenizer.get_vocab_size()
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        self.tokenizer.save(filepath)
        # Also save metadata
        metadata = {
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'eos_token': self.eos_token,
            'bos_token': self.bos_token,
            'pad_token_id': self.pad_token_id,
            'unk_token_id': self.unk_token_id,
            'eos_token_id': self.eos_token_id,
            'bos_token_id': self.bos_token_id,
        }
        metadata_path = filepath.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from file."""
        try:
            tokenizer = Tokenizer.from_file(filepath)
        except Exception as e:
            # If loading fails, it might be a corrupted file or version mismatch
            error_msg = str(e)
            if "ModelWrapper" in error_msg or "untagged enum" in error_msg:
                raise ValueError(
                    f"Tokenizer file appears corrupted or incompatible. "
                    f"This can happen due to:\n"
                    f"1. File corruption during save\n"
                    f"2. Version mismatch in tokenizers library\n"
                    f"3. Incomplete write operation\n\n"
                    f"Try deleting {filepath} and retraining the tokenizer, "
                    f"or update the tokenizers library: pip install --upgrade tokenizers\n"
                    f"Original error: {error_msg}"
                ) from e
            else:
                raise
        
        instance = cls(tokenizer)
        
        # Load metadata if available
        metadata_path = filepath.replace('.json', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                instance.pad_token = metadata['pad_token']
                instance.unk_token = metadata['unk_token']
                instance.eos_token = metadata['eos_token']
                instance.bos_token = metadata['bos_token']
                instance.pad_token_id = metadata['pad_token_id']
                instance.unk_token_id = metadata['unk_token_id']
                instance.eos_token_id = metadata['eos_token_id']
                instance.bos_token_id = metadata['bos_token_id']
        else:
            # Try to get IDs from tokenizer
            instance.pad_token_id = tokenizer.token_to_id("<PAD>")
            instance.unk_token_id = tokenizer.token_to_id("<UNK>")
            instance.eos_token_id = tokenizer.token_to_id("<EOS>")
            instance.bos_token_id = tokenizer.token_to_id("<BOS>")
        
        return instance


