import os
import shutil
import csv
from collections import defaultdict

def extract_info_from_csv(file_path):
    match_info = {}
    players_info = {}
    registry_info = {}
    other_info = {}

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0].startswith("info"):
                category = row[1]
                data = row[2:]

                if category == "team":
                    match_info[category] = match_info.get(category, []) + data
                    # Dynamically create the players_info dictionary with team names
                    for team in data:
                        players_info[team] = []
                elif category == "player":
                    team = data[0]
                    if team in players_info:
                        players_info[team].extend(data[1:])
                elif category == "registry" and row[2] == "people":
                    for i in range(1, len(data), 2):  # Assuming registry data comes in pairs of name and ID
                        player_name = data[i]
                        player_id = data[i + 1]
                        registry_info[player_name] = player_id
                else:
                    # Store any other information found under the "info" category
                    other_info[category] = data

    return match_info, players_info, registry_info, other_info

def summarize_team_matches(folder_path):
    team_stats = defaultdict(lambda: {"matches_played": 0, "matches_won": 0})
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Only process CSV files
        if file_name.endswith('.csv'):
            # Extract match information from the CSV file
            match_info, players_info, registry_info, other_info = extract_info_from_csv(file_path)
            
            if "team" in match_info:
                teams = match_info["team"]
                winner = other_info.get("winner", [None])[0]

                for team in teams:
                    team_stats[team]["matches_played"] += 1
                    if winner == team:
                        team_stats[team]["matches_won"] += 1

    return team_stats

def move_files_for_top_teams(folder_path, top_teams, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Loop through the files in the original folder and move the ones with top teams
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Only process CSV files
        if file_name.endswith('.csv'):
            match_info, players_info, registry_info, other_info = extract_info_from_csv(file_path)
            
            if "team" in match_info:
                teams = match_info["team"]
                winner = other_info.get("winner", [None])[0]
                method = other_info.get("method", [None])[0]

                # Check if the teams in the match are in the top 10 teams and exclude D/L or no winner matches
                if all(team in top_teams for team in teams) and winner is not None and method != "D/L":
                    # Copy the file to the destination folder
                    shutil.copy(file_path, os.path.join(destination_folder, file_name))

# Example usage:
folder_path = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\with_info" # Folder containing CSV files
destination_folder = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\Top_teams"  # Folder where top teams' files will be copied

# Summarize team stats
team_stats = summarize_team_matches(folder_path)

# Sort teams by the number of matches played and get the top 10
sorted_teams = sorted(team_stats.items(), key=lambda x: x[1]["matches_played"], reverse=True)
top_teams = [team[0] for team in sorted_teams[:10]]  # Extract the top 10 teams

# Copy files where only top 10 teams are playing and exclude D/L or no winner matches
move_files_for_top_teams(folder_path, top_teams, destination_folder)

# Print top 10 teams and their matches played
print("Top 10 Teams (Most Played Matches):")
for team in top_teams:
    print(f"{team}: {team_stats[team]['matches_played']} matches played")
