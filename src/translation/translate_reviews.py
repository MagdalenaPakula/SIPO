import os
import pandas as pd
from googletrans import Translator


def translate_text(text, src_lang='en', dest_lang='pl'):
    translator = Translator()
    try:
        translated = translator.translate(text, src=src_lang, dest=dest_lang)
        return translated.text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text  # Return the original text in case of error


def save_checkpoint(index):
    with open("translation_checkpoint.txt", "w") as f:
        f.write(str(index))


def load_checkpoint():
    try:
        with open("translation_checkpoint.txt", "r") as f:
            return int(f.read())
    except FileNotFoundError:
        return 0


def main():
    input_file = '../../data/raw/IMDB_Dataset.csv'
    output_file = '../../data/processed/Translated_IMDB_Dataset.csv'
    df = pd.read_csv(input_file)

    batch_size = 10
    total_rows = len(df)
    translated_rows = 0

    # Load checkpoint index
    start_index = load_checkpoint()

    if os.path.exists(output_file):
        # If the output file already exists, load it
        output_df = pd.read_csv(output_file)
    else:
        # If the output file doesn't exist, create an empty DataFrame
        output_df = pd.DataFrame(columns=df.columns)

    for i in range(start_index, total_rows, batch_size):
        batch_df = df.loc[i:i + batch_size - 1]
        batch_df['review'] = batch_df['review'].apply(translate_text)
        translated_rows += len(batch_df)
        print(f"{i}-{i + len(batch_df) - 1} rows successfully translated.")

        # Append translated batch to the output DataFrame
        output_df = pd.concat([output_df, batch_df], ignore_index=True)

        # Update checkpoint index
        save_checkpoint(i + batch_size)

        # Save the updated DataFrame to the output file
        output_df.to_csv(output_file, index=False)
        print(f"Translated batch saved to '{output_file}'.")

    print(f"Translation completed and saved to '{output_file}'.")


if __name__ == "__main__":
    main()
