import os
import shutil
import re

# Define the folder paths
top_teams_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\Top_teams"
train_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\top_teams_data_1_and_2\train"
test_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\top_teams_data_1_and_2\test"

# Create output directories for train and test splits
output_train_folder = os.path.join(top_teams_folder, "train")
output_test_folder = os.path.join(top_teams_folder, "test")
os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_test_folder, exist_ok=True)

# Helper function to extract match number from the file name
def extract_match_number(filename):
    match = re.match(r"(\d+)", filename)
    return match.group(1) if match else None

# Get match numbers from train and test folders
train_match_numbers = {
    extract_match_number(f) for f in os.listdir(train_folder) if extract_match_number(f)
}
test_match_numbers = {
    extract_match_number(f) for f in os.listdir(test_folder) if extract_match_number(f)
}

# Debug: Print match numbers
print(f"Train match numbers: {train_match_numbers}")
print(f"Test match numbers: {test_match_numbers}")

# Iterate through Top_teams folder and distribute files
for file_name in os.listdir(top_teams_folder):
    source_file = os.path.join(top_teams_folder, file_name)
    
    # Ensure the item is a file
    if not os.path.isfile(source_file):
        continue

    # Extract match number from the file name
    match_number = extract_match_number(file_name)
    
    if not match_number:
        print(f"Could not extract match number from {file_name}")
        continue

    # Check and move to the appropriate folder
    if match_number in train_match_numbers:
        shutil.move(source_file, os.path.join(output_train_folder, file_name))
        print(f"Moved {file_name} to train folder.")
    elif match_number in test_match_numbers:
        shutil.move(source_file, os.path.join(output_test_folder, file_name))
        print(f"Moved {file_name} to test folder.")
    else:
        print(f"No match found for {file_name}.")

print(f"Files have been split into:\nTrain: {output_train_folder}\nTest: {output_test_folder}")
