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


def translate(input_file, output_file):
    df = pd.read_csv(input_file)

    # Translate all rows in the 'review' column
    df['review'] = df['review'].apply(translate_text)

    # Save the translated data to the output CSV file
    df.to_csv(output_file, index=False)
    print(f"Translation completed and saved to '{output_file}'.")


def main():
    input_file = '../../data/raw/IMDB_Dataset.csv'
    output_file = '../../data/processed/Translated_IMDB_Dataset.csv'
    df = pd.read_csv(input_file)

    batch_size = 10
    total_rows = len(df)
    translated_rows = 0

    for i in range(0, total_rows, batch_size):
        batch_df = df.loc[i:i+batch_size-1]
        batch_df['review'] = batch_df['review'].apply(translate_text)
        translated_rows += len(batch_df)
        print(f"{i}-{i+len(batch_df)-1} rows successfully translated.")

    print(f"Total {translated_rows} rows translated.")
    df.to_csv(output_file, index=False)
    print(f"Translation completed and saved to '{output_file}'.")


if __name__ == "__main__":
    main()
