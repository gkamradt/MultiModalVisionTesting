from PIL import Image, ImageDraw, ImageFont
import os
import base64
import io
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv
import textwrap
import time
import json
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Levenshtein
import warnings

load_dotenv()

def levenshtein_ratio(s1, s2):
    return Levenshtein.ratio(s1, s2)

def levenshtein_distance(s1, s2):
    return Levenshtein.distance(s1, s2)

def compare_texts(original_text, returned_text):
    # Check if texts are exactly the same
    are_texts_same = original_text == returned_text

    # Calculate similarity score using difflib
    similarity_ratio = SequenceMatcher(None, original_text, returned_text).ratio()

    text_levenshtein_distance = levenshtein_distance(original_text, returned_text)
    text_levenshtein_ratio = levenshtein_ratio(original_text, returned_text)

    return are_texts_same, similarity_ratio, text_levenshtein_distance, text_levenshtein_ratio

class VisionTestingTest:
    def __init__(self, character_bounds, steps, config, distribution_type='linear', version='v1', openai_api_key=None):
        self.config = config
        self.results_file_path = config['results_file_path']
        self.image_dir = config['image_dir']
        self.background_txt_file_path = config['background_txt_file_path']
        self.character_bounds = character_bounds
        self.steps = steps
        self.distribution_type = distribution_type
        self.character_steps = self.create_distribution_from_bounds()
        self.version = version
        self.background_text = self.read_and_prepare_text(self.background_txt_file_path)
        self.processed_results = self.load_previous_results()
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

        if not self.openai_api_key:
            warnings.warn("No OPENAI_API_KEY supplied or found in environment variables.", RuntimeWarning)

    def load_previous_results(self):
        with open(self.results_file_path, 'r') as file:
            return json.load(file)
        
    def read_and_prepare_text(self, file_path):
        with open(file_path, 'r') as file:
            full_text = file.read()
        # total_words = len(full_text.split())
        return full_text
    
    # Function to convert image to base64
    def image_to_base64(self, image_path):
        with Image.open(image_path) as image:
            buffered = io.BytesIO()
            image.save(buffered, format=image.format)
            img_str = base64.b64encode(buffered.getvalue())
            return img_str.decode('utf-8')
        
    def create_distribution_from_bounds(self):
        start = self.character_bounds[0]
        end = self.character_bounds[1]

        if self.distribution_type == 'linear':
            distribution = np.linspace(start, end, num=self.steps)
            # Round the numbers to the nearest whole number and convert to integers
            distribution = np.round(distribution).astype(int)
            distribution = distribution.tolist()
        else:
            raise ValueError(f"Invalid distribution: {self.distribution_type}")
        
        return distribution

    def reset_results(self):
        with open(self.results_file_path, 'w') as file:
            json.dump([], file)

        # Delete all images
        for file in os.listdir(self.image_dir):
            os.remove(os.path.join(self.image_dir, file))

    def should_skip_combination(self, num_characters):
        for item in self.processed_results:
            if item['version_name'] == self.version and item['num_characters'] == num_characters:
                print (f"Skipping {num_characters} characters")
                return True
        return False
    
    def generate_and_save_image(self, test_text, num_characters):
        image_path = self.generate_text_image(test_text, num_characters)
        image_str = self.image_to_base64(image_path)
        return image_str, image_path
    
    def generate_text_image(self, test_text_str, num_characters):
        # Set image size, background color, and text color
        width, height = 1280, 720
        bg_color = (255, 255, 255)  # White
        text_color = (0, 0, 0)  # Black

        # Create a new image with the specified size and background color
        image = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(image)

        # For demonstration, using a default font
        font = ImageFont.load_default(size=12)

        x = (width * .01)
        y = (height * .01)

        # Use textwrap to split the text into lines that fit within the specified width
        lines = textwrap.wrap(test_text_str, width=225)  # Adjust the width parameter as needed based on the font and image size

        # Draw each line on the image, adjusting `y` for each new line
        for line in lines:
            draw.text((x, y), line, fill=text_color, font=font)
            y += 13 #font.getsize(line)[1]  # Adjust vertical position based on the height of the line

        # Save the image in the /images directory
        image_path = os.path.join(self.image_dir, f'{num_characters}_char.png')
        image.save(image_path)

        return image_path
    
    def get_results(self):
        with open(self.results_file_path, 'r') as file:
            results = json.load(file)
        return results
    
    def create_image_message(self, image_str):
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": """
You are in the mood to tell me what text you see in an image

## Instructions
* You will be given an image containing text (there is always visible text in the image I give you).
* You will not be given a blank image
* Your task is to carefully examine the image and transcribe the text contained within it exactly as it appears, preserving all spelling, punctuation, capitalization, and formatting.
* Do not correct any errors in the text or make any modifications. Only respond with the text in the image, nothing else.
* You must respond with the exact text you found in the image.
* Do not say the words "I'm sorry" or anything like that.
* Do not respond with anyting else other than the text you found in the image
* Try really hard to see the text on the image

Text you found in the image:
                     """},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_str}"
                            },
                        },
                    ]
                )
            ]
        return messages

    def get_vision_model_response(self, image_str):
        chat = ChatOpenAI(model="gpt-4-vision-preview", temperature=1, max_tokens=4096, openai_api_key=self.openai_api_key)

        messages = self.create_image_message(image_str)
        llm_response = chat.invoke(messages)
        return llm_response.content
    
    def get_background_text(self, num_characters):
        text_without_newlines = self.background_text.replace('\n', ' ').replace('\r', '')
        text_with_single_spaces = text_without_newlines.replace('.  ', '. ')
        return text_with_single_spaces[:num_characters].strip()

    def run_test(self):
        print (f"Running test for {self.config['name']} {self.version} with {self.steps} steps and {self.distribution_type} distribution")

        # Run through each step
        for num_characters in self.character_steps[:]:
            if self.should_skip_combination(num_characters):
                continue

            test_text_str = self.get_background_text(num_characters)
            image_str, image_path = self.generate_and_save_image(test_text_str, num_characters)

            start_time = time.time()
            llm_response = self.get_vision_model_response(image_str)
            end_time = time.time()
            duration = end_time - start_time

            result_data = {
                "version_name": self.version,
                "num_characters": num_characters,
                "image_path": image_path,
                "duration": duration,
                "text_submitted": test_text_str,
                "text_returned": llm_response
            }

            results = self.get_results()

            results.append(result_data)

            with open(self.results_file_path, 'w') as file:
                json.dump(results, file, indent=4)
            print (f"Finished test for {num_characters} characters in {duration} seconds")

    def process_results(self):
        results = self.get_results()

        for item in results:
            print (f"Processing {item['num_characters']} characters")
            are_texts_same, similarity_ratio, text_levenshtein_distance, text_levenshtein_ratio = compare_texts(item['text_submitted'], item['text_returned'])

            # Append new values to the item dictionary
            item['are_texts_same'] = are_texts_same
            item['similarity_ratio'] = similarity_ratio
            item['text_levenshtein_distance'] = text_levenshtein_distance
            item['text_levenshtein_ratio'] = text_levenshtein_ratio

        # Sort the results by 'num_characters' key before saving
        sorted_results = sorted(results, key=lambda x: x['num_characters'])

        with open(self.results_file_path, 'w') as file:
            json.dump(sorted_results, file, indent=4)

    def visualize_results(self):
        # Ensure the visualizations directory exists
        os.makedirs('visualizations', exist_ok=True)

        # Initialize the plot
        plt.figure(figsize=(10, 6))

        # List all files in the results directory
        results_files = os.listdir('results')
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a list of colors for the lines

        for i, file_name in enumerate(results_files):
            # Construct the full file path
            file_path = os.path.join('results', file_name)

            # Load the JSON data into a DataFrame
            df = pd.read_json(file_path)
            df.sort_values(by='num_characters', inplace=True)

            # Plotting each file's data on a separate line
            plt.plot(df['num_characters'], df['text_levenshtein_ratio'] * 100, marker='o', linestyle='-', color=colors[i % len(colors)], label=file_name)  # Cycle through colors

        # Plot formatting
        plt.title('Levenshtein Ratio by Number of Characters')
        plt.ylim(0, 101)
        plt.xlabel('Number of Characters')
        plt.ylabel('Levenshtein Ratio (%)')
        plt.grid(True)
        plt.legend(title='Results File', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Format y-axis ticks as percentages
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter())

        # Save the plot as a PNG file
        plt.savefig('visualizations/similarity_ratio_vs_num_characters.png', bbox_inches='tight')
        # Optionally, show the plot
        # plt.show()

# Ensure to call the main function
if __name__ == "__main__":
    config = {
        'pg_essay' : {
            'name' : 'PG Essay',
            'image_dir': 'images/images_01_pg_essay',
            'results_file_path' : 'results/results_01.json',
            'background_txt_file_path' : 'background_text/bg_txt_01_pg_essay.txt',
        },
        'pg_essay_randomized' : {
            'name' : 'PG Essay (Randomized)',
            'image_dir': 'images/images_02_pg_essay_randomized',
            'results_file_path' : 'results/results_02_randomized.json',
            'background_txt_file_path' : 'background_text/bg_txt_02_pg_essay_randomized.txt',
        },
        'random_tokens' : {
            'name' : 'Random Tokens',
            'image_dir': 'images/images_03_random_tokens',
            'results_file_path' : 'results/results_03_random_tokens.json',
            'background_txt_file_path' : 'background_text/bg_txt_03_random_tokens.txt',
        },
        'random_characters' : {
            'name' : 'Random Characters',
            'image_dir': 'images/images_04_random_characters',
            'results_file_path' : 'results/results_04_random_characters.json',
            'background_txt_file_path' : 'background_text/bg_txt_04_random_characters.txt',
        }
    }

    for test in config:
        print (f"Running test for {test}")
        test_config = config[test]

        vtt = VisionTestingTest(character_bounds=(10,10000),
                                steps=30,
                                config=test_config)
        # vtt.reset_results()

        vtt.run_test()

        vtt.process_results()

        vtt.visualize_results()