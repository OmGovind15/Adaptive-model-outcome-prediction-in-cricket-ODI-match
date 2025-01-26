import os
import shutil

# Define the directories
dir1 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\Top_teams"
dir2 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\without_info"
output_dir = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\top_teams_data_1_and_2"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get the list of files in both directories
files_dir1 = os.listdir(dir1)
files_dir2 = os.listdir(dir2)

# Loop through the files in dir1 and match based on numbers in filenames
for file1 in files_dir1:
    # Extract the number from the filename (assuming the number is at the start)
    file1_number = file1.split('_')[0]  # Split based on underscore and get the first part
    for file2 in files_dir2:
        file2_number = file2.split('.')[0]  # Get the number part of the file2 name
        if file1_number == file2_number:
            # If numbers match, copy file2 to the output directory
            shutil.copy(os.path.join(dir2, file2), os.path.join(output_dir, file2))
            print(f"Copied {file2} to {output_dir}")

print("Task completed!")
