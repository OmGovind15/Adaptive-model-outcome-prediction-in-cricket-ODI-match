import os
import pandas as pd
from tqdm import tqdm

def process_match_data(folder_path_innings1, folder_path_innings2, stats_file_path):
    """
    Process match data from separate folders for innings 1 and 2, merge with stats file,
    and compute weighted averages for each match.
    
    Parameters:
        folder_path_innings1 (str): Path to the folder containing match data for innings 1.
        folder_path_innings2 (str): Path to the folder containing match data for innings 2.
        stats_file_path (str): Path to the CSV file with player statistics, including innings 1 and innings 2 data.
    
    Returns:
        dict: Contains unique players not found in both innings.
    """
    
    # Load the stats file containing separate columns for innings 1 and 2
    player_stats = pd.read_csv(stats_file_path)
    
    def get_batting_average(player, innings):
        player_data = player_stats[player_stats['player'] == player]
        if not player_data.empty:
            runs = player_data[f'total_runs_{innings}'].values[0]
            matches = player_data[f'matches_played_{innings}'].values[0]
            return runs / matches if matches > 0 else 0
        else:
            return 0

    def get_bowling_average(player, innings):
        player_data = player_stats[player_stats['player'] == player]
        if not player_data.empty:
            wickets = player_data[f'total_wickets_{innings}'].values[0]
            matches = player_data[f'matches_played_{innings}'].values[0]  # Matches played instead of balls bowled
            return wickets / matches if matches > 0 else 0
        else:
            return 0

    def get_weighted_batting_average(batting_bowler_list, balls_faced_list, innings):
        total_weighted_runs = 0
        total_balls = 0
        not_found_players = set()

        for batter, balls_faced in zip(batting_bowler_list, balls_faced_list):
            batting_avg = get_batting_average(batter, innings)
            if batting_avg is None:
                not_found_players.add(batter)
                batting_avg = 0
            total_weighted_runs += batting_avg * balls_faced
            total_balls += balls_faced

        weighted_batting_avg = total_weighted_runs / total_balls if total_balls > 0 else 0
        return weighted_batting_avg, not_found_players

    def get_weighted_bowling_average(batting_bowler_list, balls_bowled_list, innings):
        total_weighted_bowling_avg = 0
        total_balls = 0
        not_found_players = set()

        for bowler, balls_bowled in zip(batting_bowler_list, balls_bowled_list):
            bowling_avg = get_bowling_average(bowler, innings)
            if bowling_avg is None:
                not_found_players.add(bowler)
                bowling_avg = 0
            total_weighted_bowling_avg += bowling_avg * balls_bowled
            total_balls += balls_bowled

        weighted_bowling_avg = total_weighted_bowling_avg / total_balls if total_balls > 0 else 0
        return weighted_bowling_avg, not_found_players

    # Unique player sets for players not found in stats file for both innings
    unique_not_found_batting_1 = set()
    unique_not_found_bowling_1 = set()
    unique_not_found_batting_2 = set()
    unique_not_found_bowling_2 = set()

    # List all files in the respective folders for innings 1 and innings 2
    files_innings1 = [filename for filename in os.listdir(folder_path_innings1) if filename.endswith(".csv")]
    files_innings2 = [filename for filename in os.listdir(folder_path_innings2) if filename.endswith(".csv")]

    # Process innings 1 files
    for filename in tqdm(files_innings1, desc="Processing innings 1 files", unit="file"):
        file_path = os.path.join(folder_path_innings1, filename)

        # Load the innings 1 data
        innings1_data = pd.read_csv(file_path)

        balls_faced_list_innings1 = [
            list(map(int, balls.split(', '))) if isinstance(balls, str) else [balls]
            for balls in innings1_data['balls_faced']
        ]
        balls_bowled_list_innings1 = [
            list(map(int, balls.split(', '))) if isinstance(balls, str) else [balls]
            for balls in innings1_data['balls_bowled']
        ]

        # Add weighted batting and bowling columns for innings 1
        innings1_data['weighted_batting_average'], not_found_batting_1 = zip(*innings1_data.apply(
            lambda row: get_weighted_batting_average(row['batters'].split(', '), balls_faced_list_innings1[innings1_data.index.get_loc(row.name)], 'innings_1'),
            axis=1
        ))

        innings1_data['weighted_bowling_average'], not_found_bowling_1 = zip(*innings1_data.apply(
            lambda row: get_weighted_bowling_average(row['bowlers'].split(', '), balls_bowled_list_innings1[innings1_data.index.get_loc(row.name)], 'innings_1'),
            axis=1
        ))

        # Update unique not found players for innings 1
        for players in not_found_batting_1:
            unique_not_found_batting_1.update(players)
        for players in not_found_bowling_1:
            unique_not_found_bowling_1.update(players)

        # Save the updated innings 1 data
        innings1_data.to_csv(file_path, index=False)

    # Process innings 2 files
    for filename in tqdm(files_innings2, desc="Processing innings 2 files", unit="file"):
        file_path = os.path.join(folder_path_innings2, filename)

        # Load the innings 2 data
        innings2_data = pd.read_csv(file_path)

        balls_faced_list_innings2 = [
            list(map(int, balls.split(', '))) if isinstance(balls, str) else [balls]
            for balls in innings2_data['balls_faced']
        ]
        balls_bowled_list_innings2 = [
            list(map(int, balls.split(', '))) if isinstance(balls, str) else [balls]
            for balls in innings2_data['balls_bowled']
        ]

        # Add weighted batting and bowling columns for innings 2
        innings2_data['weighted_batting_average'], not_found_batting_2 = zip(*innings2_data.apply(
            lambda row: get_weighted_batting_average(row['batters'].split(', '), balls_faced_list_innings2[innings2_data.index.get_loc(row.name)], 'innings_2'),
            axis=1
        ))

        innings2_data['weighted_bowling_average'], not_found_bowling_2 = zip(*innings2_data.apply(
            lambda row: get_weighted_bowling_average(row['bowlers'].split(', '), balls_bowled_list_innings2[innings2_data.index.get_loc(row.name)], 'innings_2'),
            axis=1
        ))

        # Update unique not found players for innings 2
        for players in not_found_batting_2:
            unique_not_found_batting_2.update(players)
        for players in not_found_bowling_2:
            unique_not_found_bowling_2.update(players)

        # Save the updated innings 2 data
        innings2_data.to_csv(file_path, index=False)

    # Return summary with lists of not found players for both innings
    return {
        "unique_not_found_batting_innings_1": len(unique_not_found_batting_1),
        "unique_not_found_bowling_innings_1": len(unique_not_found_bowling_1),
        "unique_not_found_batting_innings_2": len(unique_not_found_batting_2),
        "unique_not_found_bowling_innings_2": len(unique_not_found_bowling_2),
        "not_found_batting_players_innings_1": list(unique_not_found_batting_1),
        "not_found_bowling_players_innings_1": list(unique_not_found_bowling_1),
        "not_found_batting_players_innings_2": list(unique_not_found_batting_2),
        "not_found_bowling_players_innings_2": list(unique_not_found_bowling_2)
    }


folder_path_innings1 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\train_processed"
folder_path_innings2 = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\train_processed"
stats_file_path = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\modified_file.csv"

result = process_match_data(folder_path_innings1, folder_path_innings2, stats_file_path)
result_2=process_match_data(r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings1\test_processed", r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test_processed",stats_file_path)
print(f"Unique players not found for batting in innings 1: {result['unique_not_found_batting_innings_1']}")
print(f"Unique players not found for bowling in innings 1: {result['unique_not_found_bowling_innings_1']}")
print("Players not found for batting in innings 1:")
for player in result["not_found_batting_players_innings_1"]:
    print(player)

print(f"Unique players not found for batting in innings 2: {result['unique_not_found_batting_innings_2']}")
print(f"Unique players not found for bowling in innings 2: {result['unique_not_found_bowling_innings_2']}")
print("Players not found for batting in innings 2:")
for player in result["not_found_batting_players_innings_2"]:
    print(player)
