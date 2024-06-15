import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from googletrans import Translator

# Define PATHS
input_file = '../../data/raw/IMDB_Dataset.csv'
output_file = '../../data/processed/Translated_IMDB_Dataset2.csv'
checkpoint_file = 'translation_checkpoint.txt'
batch_size = 25


def translate_text(text, src_lang='en', dest_lang='pl'):
    translator = Translator()
    try:
        translated = translator.translate(text, src=src_lang, dest=dest_lang)
        return translated.text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text  # Return the original text in case of error


def save_checkpoint(index):
    with open(checkpoint_file, "w") as f:
        f.write(str(index))


def load_checkpoint():
    try:
        with open(checkpoint_file, "r") as f:
            return int(f.read())
    except FileNotFoundError:
        return 0


def translate_batch(df, start_index, batch_size):
    batch_df = df.loc[start_index:start_index + batch_size - 1].copy()
    batch_df.loc[:, 'review'] = batch_df['review'].apply(translate_text)
    return batch_df


def load_data(input_file):
    return pd.read_csv(input_file)


def save_translated_batch(output_file, batch_df, append=True):
    if append and os.path.exists(output_file):
        batch_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        batch_df.to_csv(output_file, index=False)


def load_output_data(output_file, df_columns):
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    else:
        return pd.DataFrame(columns=df_columns)


def process_translation_batches(df, start_index, total_rows, batch_size, output_file):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(translate_batch, df, i, batch_size) for i in
                   range(start_index, total_rows, batch_size)]
        for future in futures:
            batch_df = future.result()
            print(f"{batch_df.index[0]}-{batch_df.index[-1]} rows successfully translated.")
            save_translated_batch(output_file, batch_df)
            save_checkpoint(batch_df.index[-1] + 1)
            print(f"Translated batch saved to '{output_file}'.")


def main():
    global input_file, output_file, checkpoint_file, batch_size

    df = load_data(input_file)
    total_rows = len(df)
    start_index = load_checkpoint()

    if os.path.exists(output_file):
        output_df = load_output_data(output_file, df.columns)
    else:
        output_df = pd.DataFrame(columns=df.columns)

    process_translation_batches(df, start_index, total_rows, batch_size, output_file)

    print(f"Translation completed and saved to '{output_file}'.")


if __name__ == "__main__":
    main()
