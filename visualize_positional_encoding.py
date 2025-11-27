"""
Visualize positional encoding to understand how it indicates position.

Usage:
    python visualize_positional_encoding.py
"""

import torch
import math
import numpy as np
import matplotlib.pyplot as plt


def create_positional_encoding(d_model=512, max_seq_len=10):
    """
    Create positional encoding using the same formula as model.py
    
    Returns:
        pe: [max_seq_len, d_model] positional encoding matrix
    """
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                       (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


def visualize_encoding_values():
    """Show actual positional encoding values for first few positions."""
    print("=" * 80)
    print("POSITIONAL ENCODING VALUES")
    print("=" * 80)
    
    # Use smaller d_model for readability
    d_model = 8
    max_seq_len = 5
    
    pe = create_positional_encoding(d_model, max_seq_len)
    
    print(f"\nConfiguration: d_model={d_model}, max_seq_len={max_seq_len}")
    print(f"\nPositional encoding matrix shape: {pe.shape}")
    print(f"\nFirst 8 dimensions for positions 0-4:\n")
    
    # Print table
    print(f"{'Position':<10}", end="")
    for dim in range(d_model):
        print(f"Dim {dim:<8}", end="")
    print()
    print("-" * 80)
    
    for pos in range(max_seq_len):
        print(f"{pos:<10}", end="")
        for dim in range(d_model):
            value = pe[pos, dim].item()
            print(f"{value:8.4f}  ", end="")
        print()
    
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS:")
    print("=" * 80)
    print("1. Each position has a UNIQUE pattern of values")
    print("2. Even dimensions (0, 2, 4, 6) use sin()")
    print("3. Odd dimensions (1, 3, 5, 7) use cos()")
    print("4. Lower dimensions change more rapidly")
    print("5. Higher dimensions change more slowly")
    print()


def demonstrate_position_distinction():
    """Demonstrate how positional encoding distinguishes positions."""
    print("=" * 80)
    print("HOW POSITIONAL ENCODING DISTINGUISHES POSITIONS")
    print("=" * 80)
    
    d_model = 8
    pe = create_positional_encoding(d_model, 3)
    
    # Simulate word embeddings
    word_embedding = torch.tensor([0.5, 0.3, -0.2, 0.4, 0.1, -0.3, 0.2, 0.0])
    
    print(f"\nExample word embedding: {word_embedding.numpy()}")
    print(f"\nSame word at different positions:\n")
    
    for pos in range(3):
        combined = word_embedding + pe[pos]
        print(f"Position {pos}:")
        print(f"  Word embedding:  {word_embedding.numpy()}")
        print(f"  Positional PE:   {pe[pos].numpy()}")
        print(f"  Combined result: {combined.numpy()}")
        print()
    
    print("=" * 80)
    print("KEY POINT:")
    print("=" * 80)
    print("The SAME word at DIFFERENT positions produces DIFFERENT vectors!")
    print("This allows the model to distinguish word order.")
    print()


def show_relative_positions():
    """Show how relative positions are encoded."""
    print("=" * 80)
    print("RELATIVE POSITION ENCODING")
    print("=" * 80)
    
    d_model = 8
    pe = create_positional_encoding(d_model, 5)
    
    print("\nPositional encodings for positions 0-4:")
    print("(Showing first 4 dimensions for clarity)\n")
    
    print(f"{'Position':<10} {'Dim 0':<12} {'Dim 1':<12} {'Dim 2':<12} {'Dim 3':<12}")
    print("-" * 60)
    
    for pos in range(5):
        dim0 = pe[pos, 0].item()
        dim1 = pe[pos, 1].item()
        dim2 = pe[pos, 2].item()
        dim3 = pe[pos, 3].item()
        print(f"{pos:<10} {dim0:12.6f} {dim1:12.6f} {dim2:12.6f} {dim3:12.6f}")
    
    print("\n" + "=" * 80)
    print("OBSERVATION:")
    print("=" * 80)
    print("The encoding naturally captures relative distances.")
    print("For example, position 1 and position 2 are 1 position apart,")
    print("and this relationship is encoded in the PE values.")
    print()


def example_sentence():
    """Show complete example with a sentence."""
    print("=" * 80)
    print("COMPLETE EXAMPLE: 'The cat sat'")
    print("=" * 80)
    
    d_model = 8
    pe = create_positional_encoding(d_model, 3)
    
    # Simulate embeddings for "The", "cat", "sat"
    embeddings = {
        "The": torch.tensor([0.1, 0.3, -0.2, 0.5, 0.0, 0.2, -0.1, 0.1]),
        "cat": torch.tensor([0.4, -0.1, 0.6, -0.3, 0.2, 0.1, -0.2, 0.0]),
        "sat": torch.tensor([-0.2, 0.5, 0.1, 0.4, -0.1, 0.3, 0.0, -0.1])
    }
    
    sentence = ["The", "cat", "sat"]
    
    print(f"\nSentence: {' '.join(sentence)}")
    print(f"\nStep-by-step with positional encoding:\n")
    
    for i, word in enumerate(sentence):
        word_emb = embeddings[word]
        pos_emb = pe[i]
        combined = word_emb + pos_emb
        
        print(f"Position {i}: '{word}'")
        print(f"  Word embedding:  {word_emb.numpy()}")
        print(f"  Positional PE:   {pos_emb.numpy()}")
        print(f"  Combined:        {combined.numpy()}")
        print()
    
    print("=" * 80)
    print("RESULT:")
    print("=" * 80)
    print("Each word now has BOTH semantic information (from embedding)")
    print("AND position information (from positional encoding).")
    print("This allows the model to distinguish:")
    print("  - 'cat' at position 0 vs position 1")
    print("  - Word order matters!")
    print()


def compare_sentences():
    """Compare how same words in different orders are distinguished."""
    print("=" * 80)
    print("COMPARING: 'The cat sat' vs 'cat The sat'")
    print("=" * 80)
    
    d_model = 8
    pe = create_positional_encoding(d_model, 3)
    
    embeddings = {
        "The": torch.tensor([0.1, 0.3, -0.2, 0.5, 0.0, 0.2, -0.1, 0.1]),
        "cat": torch.tensor([0.4, -0.1, 0.6, -0.3, 0.2, 0.1, -0.2, 0.0]),
        "sat": torch.tensor([-0.2, 0.5, 0.1, 0.4, -0.1, 0.3, 0.0, -0.1])
    }
    
    sentence1 = ["The", "cat", "sat"]
    sentence2 = ["cat", "The", "sat"]
    
    print(f"\nSentence 1: {' '.join(sentence1)}")
    print(f"Sentence 2: {' '.join(sentence2)}")
    print(f"\nWithout positional encoding: These would be IDENTICAL!")
    print(f"\nWith positional encoding:\n")
    
    print("Sentence 1:")
    for i, word in enumerate(sentence1):
        combined = embeddings[word] + pe[i]
        print(f"  Pos {i} ('{word}'): {combined.numpy()}")
    
    print("\nSentence 2:")
    for i, word in enumerate(sentence2):
        combined = embeddings[word] + pe[i]
        print(f"  Pos {i} ('{word}'): {combined.numpy()}")
    
    print("\n" + "=" * 80)
    print("KEY DIFFERENCE:")
    print("=" * 80)
    print("In Sentence 1: 'cat' is at position 1 → embedding('cat') + PE(1)")
    print("In Sentence 2: 'cat' is at position 0 → embedding('cat') + PE(0)")
    print("\nThese are DIFFERENT vectors! The model can distinguish word order.")
    print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("POSITIONAL ENCODING VISUALIZATION")
    print("=" * 80)
    print("\nThis script demonstrates how positional encoding indicates position")
    print("in transformer models.\n")
    
    visualize_encoding_values()
    demonstrate_position_distinction()
    show_relative_positions()
    example_sentence()
    compare_sentences()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Positional encoding indicates position by:

1. Creating UNIQUE patterns for each position using sin/cos functions
2. Adding these patterns to word embeddings
3. Making same words DIFFERENT when at different positions
4. Encoding relative positions through mathematical structure
5. Using multiple frequencies to capture different position scales

Result: The model can distinguish word order and learn position-dependent
relationships, which is essential for understanding language!
    """)


if __name__ == '__main__':
    main()



