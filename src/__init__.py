import os
import pandas as pd
from googletrans import Translator
from concurrent.futures import ThreadPoolExecutor


# Function to translate text
def translate_text(text, src_lang='en', dest_lang='pl'):
    translator = Translator()
    try:
        translated = translator.translate(text, src=src_lang, dest=dest_lang)
        return translated.text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text  # Return the original text in case of error


# Function to split text into chunks
def split_text(text, chunk_size=5000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# Function to translate a row which has more than 5k chars
def translate_row(row):
    if len(row['review']) > 5000:
        # Split the review text into chunks of 5000 characters or less
        chunks = split_text(row['review'])
        # Translate each chunk individually
        translated_chunks = [translate_text(chunk) for chunk in chunks]
        # Join the translated chunks
        translated_text = ''.join(translated_chunks)
        # Update the 'review' column with the translated text
        row['review'] = translated_text
    return row


