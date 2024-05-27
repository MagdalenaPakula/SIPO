import pandas as pd

input_path = '../../data/raw/IMDB_Dataset.csv'

# Load the original CSV file
data = pd.read_csv(input_path)

# Load the original CSV file
data = pd.read_csv(input_path)

# Determine the number of rows
total_rows = len(data)
print(f"Total rows: {total_rows}")

# Calculate the split index
split_index = total_rows // 2

# Split the data
data_1half = data.iloc[:split_index]
data_2half = data.iloc[split_index:]

# Save the first half to data_1half.csv
data_1half.to_csv('data_1half.csv', index=False)

# Save the second half to data_2half.csv
data_2half.to_csv('data_2half.csv', index=False)

print("Data has been split and saved into data_1half.csv and data_2half.csv")

