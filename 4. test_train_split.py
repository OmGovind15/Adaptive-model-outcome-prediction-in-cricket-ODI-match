import os
import shutil
import random

# Path to your folder containing the CSV files
source_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\top_teams_data_1_and_2"

# Subfolders where the files will be moved
train_folder = os.path.join(source_folder, "train")
test_folder = os.path.join(source_folder, "test")

# Create the subfolders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get the list of all CSV files in the source folder
csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

# Shuffle the list of CSV files to randomize the split
random.shuffle(csv_files)

# Split the files into train and test sets (70% train, 30% test)
split_index = int(0.7 * len(csv_files))
train_files = csv_files[:split_index]
test_files = csv_files[split_index:]

# Move the files into the respective subfolders
for file in train_files:
    shutil.move(os.path.join(source_folder, file), os.path.join(train_folder, file))

for file in test_files:
    shutil.move(os.path.join(source_folder, file), os.path.join(test_folder, file))

print("Files have been successfully distributed into train and test folders.")
