import random
import string

# Function to generate 10000 random characters including whitespace
def generate_random_characters():
    characters = string.ascii_letters + string.digits + string.punctuation + ' ' * 10 + '\n' * 2  # Increase whitespace probability
    return ''.join(random.choice(characters) for _ in range(10000))

# Generate the characters
random_characters = generate_random_characters()

# Save to a file
with open('background_text/bg_txt_04_random_characters.txt', 'w') as file:
    file.write(random_characters)

print("File saved as bg_txt_04_random_characters.txt")