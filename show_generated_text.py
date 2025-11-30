#!/usr/bin/env python3
"""
Extract and display only the generated text from the generation output.
"""

import subprocess
import re
import sys

def extract_generated_text(output):
    """Extract generated text from output"""
    # Pattern to match "Generated text:" followed by the text
    pattern = r'Generated text:\s*\n(.*?)(?=\n\n|\nUsing checkpoint|$)'
    matches = re.findall(pattern, output, re.DOTALL)
    return matches

def main():
    num_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    prompt_length = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    max_length = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
    
    # Run generation
    cmd = ["./generate_training_prompt.sh", str(num_prompts), str(prompt_length), str(max_length), str(temperature)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract prompts and generated text
    output = result.stdout + result.stderr
    
    # Extract prompts
    prompt_pattern = r'PROMPT \(from training data\): (.*?)(?=\n|$)'
    prompts = re.findall(prompt_pattern, output)
    
    # Extract generated text
    generated_pattern = r'Generated text:\s*\n(.*?)(?=\n\n|\nUsing checkpoint|$)'
    generated_texts = re.findall(generated_pattern, output, re.DOTALL)
    
    # Display results
    print("=" * 80)
    print("GENERATED TEXT OUTPUTS")
    print("=" * 80)
    print()
    
    for i, (prompt, generated) in enumerate(zip(prompts, generated_texts), 1):
        print(f"{'─' * 80}")
        print(f"GENERATION {i}/{len(prompts)}")
        print(f"{'─' * 80}")
        print(f"\n📝 Prompt: {prompt}")
        print(f"\n✨ Generated Text:\n")
        # Clean up the generated text
        text = generated.strip()
        # Remove excessive "The end" repetitions (keep max 2)
        text = re.sub(r'(The end \. ){3,}', 'The end. ', text)
        print(text)
        print("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    main()



