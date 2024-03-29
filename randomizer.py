import random
import tiktoken

def randomize_words_in_file(input_file_path, output_file_path):
    # Read the contents of the input file
    with open(input_file_path, 'r') as file:
        words = file.read().split()
    
    # Randomize the order of the words
    random.shuffle(words)
    
    # Write the randomized words to the output file
    with open(output_file_path, 'w') as file:
        file.write(' '.join(words))

# Specify the input and output file paths
input_file_path = 'pg_essay.txt'
output_file_path = 'pg_essay_word_randomized.txt'


def random_token_text_file(size=10000):
    # Get the tokenizer for GPT-4
    enc = tiktoken.get_encoding("cl100k_base")
    max_token_value = enc.max_token_value

    # Generate 3000 random tokens. Assuming the vocabulary size is large, adjust if needed.
    # Note: The actual vocabulary size for GPT-4 is not specified in the provided context,
    # so you might need to adjust the range based on the tokenizer's properties.
    random_tokens = [random.randint(1, max_token_value - 10) for _ in range(size)]

    # Loop through all the tokens, print them out before you try and encode them then try and encode them
    for token in random_tokens:
        print(token)
        print(enc.decode([token]))

    # Decode the tokens back to text
    random_text = enc.decode(random_tokens)

    # Write the text to a file
    with open("random_tokens.txt", "w", encoding="utf-8") as file:
        file.write(random_text)

size = 10000
random_token_text_file(size)
print(f"Generated text with {size} random tokens saved to random_tokens.txt")


# Call the function to randomize words in the file
# randomize_words_in_file(input_file_path, output_file_path)