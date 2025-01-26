import os
import csv
from collections import defaultdict

def extract_team_info_from_csv(file_path):
    """Extract team information and match winner from a single CSV file."""
    teams = []
    winner = None
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0].startswith("info") and row[1] == "team":
                teams.append(row[2])
            if row[0].startswith("info") and row[1] == "winner":
                winner = row[2]  # Winner is the team name
    return teams, winner

def count_matches_by_team(folder_path):
    """Count the number of matches played and matches won by each team across multiple CSV files."""
    team_match_count = defaultdict(int)
    team_win_count = defaultdict(int)
    head_to_head_count = defaultdict(lambda: defaultdict(int))
    head_to_head_win = defaultdict(lambda: defaultdict(int))

    for file_name in os.listdir(folder_path):
        if file_name.endswith("_info.csv"):  # Filter for relevant files
            file_path = os.path.join(folder_path, file_name)
            teams, winner = extract_team_info_from_csv(file_path)

            # Increment match count for each team
            for team in teams:
                team_match_count[team] += 1
                if team == winner:
                    team_win_count[team] += 1

            # Head-to-head match count
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team1, team2 = teams[i], teams[j]
                    head_to_head_count[team1][team2] += 1
                    head_to_head_count[team2][team1] += 1

                    if winner == team1:
                        head_to_head_win[team1][team2] += 1
                    elif winner == team2:
                        head_to_head_win[team2][team1] += 1

    return team_match_count, team_win_count, head_to_head_count, head_to_head_win

def calculate_winning_percentages(team_match_count, team_win_count):
    """Calculate overall winning percentage for each team."""
    winning_percentages = {}
    for team in team_match_count:
        if team_match_count[team] > 0:
            winning_percentages[team] = (team_win_count[team] / team_match_count[team]) * 100
    return winning_percentages

def calculate_head_to_head_percentages(head_to_head_count, head_to_head_win):
    """Calculate head-to-head winning percentage for each pair of teams."""
    head_to_head_percentages = {}
    for team1 in head_to_head_count:
        for team2 in head_to_head_count[team1]:
            if team1 != team2:  # Only calculate for distinct pairs
                total_matches = head_to_head_count[team1][team2]
                if total_matches > 0:  # Ensure there's at least one match
                    win1 = head_to_head_win[team1][team2]
                    win2 = head_to_head_win[team2][team1]
                    win1_percentage = (win1 / total_matches) * 100
                    win2_percentage = (win2 / total_matches) * 100
                    head_to_head_percentages[(team1, team2)] = (win1_percentage, win2_percentage)
    return head_to_head_percentages

def add_win_percentage_column_to_csv(input_folder, head_to_head_percentages):
    """Add head-to-head win percentage for the bowling team in each CSV file."""
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(input_folder, file_name)
            updated_rows = []
            
            # Read the CSV file
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                fieldnames = reader.fieldnames + ['bowling_team_win_percentage']
                for row in reader:
                    bowling_team = row['bowling_team']
                    batting_team = row['batting_team']
                    # Get the head-to-head win percentage for the bowling team
                    if (bowling_team, batting_team) in head_to_head_percentages:
                        win_percentage = head_to_head_percentages[(bowling_team, batting_team)][0]  # Bowling team's win percentage
                    else:
                        win_percentage = 0.0  # If no matches found, set win percentage to 0
                    row['bowling_team_win_percentage'] = win_percentage
                    updated_rows.append(row)

            # Write the updated CSV file with the new column
            with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(updated_rows)

folder_path_top_10_teams = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\Top_teams\train" # Path to the folder containing the match CSVs
folder_path_final_dataset = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test_processed" # Path to the final dataset files

# Get the match data
team_match_count, team_win_count, head_to_head_count, head_to_head_win = count_matches_by_team(folder_path_top_10_teams)

# Calculate head-to-head percentages
head_to_head_percentages = calculate_head_to_head_percentages(head_to_head_count, head_to_head_win)

# Add the win percentage column to each CSV file in the final dataset folder
add_win_percentage_column_to_csv(folder_path_final_dataset, head_to_head_percentages)

print("Win percentage column added to CSV files.")
