import os
import pandas as pd
from tqdm import tqdm  # Import tqdm

# Folder paths
input_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\top_teams_data_1_and_2\test"  # Path to the folder containing the CSV files
innings1_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test"  # Path to store inning 1 files
innings2_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test"  # Path to store inning 2 files

# Create directories if they don't exist
os.makedirs(innings1_folder, exist_ok=True)
os.makedirs(innings2_folder, exist_ok=True)

# Get all CSV files in the input folder
csv_files = [filename for filename in os.listdir(input_folder) if filename.endswith('.csv')]

# Loop through all CSV files in the input folder with tqdm progress bar
for filename in tqdm(csv_files, desc="Processing files", unit="file"):
    file_path = os.path.join(input_folder, filename)
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Filter rows for innings 1 and innings 2
    df_innings1 = df[df['innings'] == 1]
    df_innings2 = df[df['innings'] == 2]
    
    # Define file paths for the new innings CSVs
    innings1_file = os.path.join(innings1_folder, f'innings1_{filename}')
    innings2_file = os.path.join(innings2_folder, f'innings2_{filename}')
    
    # Save the filtered DataFrames to separate files
    df_innings1.to_csv(innings1_file, index=False)
    df_innings2.to_csv(innings2_file, index=False)

    print(f"Processed {filename}: Divided into innings1 and innings2.")
