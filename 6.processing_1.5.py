import pandas as pd
import os
from tqdm import tqdm

def process_batting_bowling_data(folder_path, player_stats_path):
    """
    Process batting and bowling data to compute weighted batting and bowling averages,
    updating the data directly in the existing files.

    Parameters:
        folder_path (str): Path to the folder containing CSV files with batting-bowling data.
        player_stats_path (str): Path to the CSV file with player statistics.

    Returns:
        dict: Contains total unique counts of players not found for batting and bowling,
              along with the lists of player names not found.
    """
    # Load the player stats data
    player_stats = pd.read_csv(player_stats_path)

    # Create a function to get the batting average for a player
    def get_batting_average(player):
        player_data = player_stats[player_stats['player'] == player]
        if not player_data.empty:
            runs = player_data['total_runs_innings_1'].values[0]
            matches = player_data['matches_played_innings_1'].values[0]
            return runs / matches if matches > 0 else 0
        else:
            return 0

    # Create a function to calculate the weighted batting average
    def get_weighted_batting_average(batting_bowler_list, balls_faced_list):
        total_weighted_runs = 0
        total_balls = 0
        not_found_players = set()

        for batter, balls_faced in zip(batting_bowler_list, balls_faced_list):
            batting_avg = get_batting_average(batter)
            if batting_avg is None:
                not_found_players.add(batter)
                batting_avg = 0
            total_weighted_runs += batting_avg * balls_faced  # Weighted by balls faced
            total_balls += balls_faced  # Sum of all balls faced

        weighted_batting_avg = total_weighted_runs / total_balls if total_balls > 0 else 0
        return weighted_batting_avg, not_found_players

    # Create a function to calculate individual bowling average
    def get_bowling_average(player):
        player_data = player_stats[player_stats['player'] == player]
        if not player_data.empty:
            wickets = player_data['total_wickets_innings_1'].values[0]
            matches = player_data['matches_played_innings_1'].values[0]  # Using matches played instead of balls bowled
            return wickets / matches if matches > 0 else 0
        else:
            return 0

    # Create a function to calculate the weighted bowling average
    def get_weighted_bowling_average(batting_bowler_list, balls_bowled_list):
        total_weighted_bowling_avg = 0
        total_balls = 0
        not_found_players = set()

        for bowler, balls_bowled in zip(batting_bowler_list, balls_bowled_list):
            bowling_avg = get_bowling_average(bowler)
            if bowling_avg is None:
                not_found_players.add(bowler)
                bowling_avg = 0
            total_weighted_bowling_avg += bowling_avg * balls_bowled  # Weighted by balls bowled
            total_balls += balls_bowled  # Sum of all balls bowled

        weighted_bowling_avg = total_weighted_bowling_avg / total_balls if total_balls > 0 else 0
        return weighted_bowling_avg, not_found_players

    # Initialize sets for unique players not found
    unique_not_found_batting = set()
    unique_not_found_bowling = set()

    # List all CSV files in the folder
    files = [filename for filename in os.listdir(folder_path) if filename.endswith(".csv")]

    # Process each file
    for filename in tqdm(files, desc="Processing files", unit="file"):
        file_path = os.path.join(folder_path, filename)

        # Load the batting-bowling data
        batting_bowling_data = pd.read_csv(file_path)

        # Parse balls faced and balls bowled (handling both string and integer values)
        def parse_balls(balls_column):
            return [
                list(map(int, balls.split(', '))) if isinstance(balls, str) else [balls]
                for balls in balls_column
            ]

        balls_faced_list = parse_balls(batting_bowling_data['balls_faced'])
        balls_bowled_list = parse_balls(batting_bowling_data['balls_bowled'])

        # Add columns for weighted batting and bowling averages
        batting_bowling_data['weighted_batting_average'], not_found_batting = zip(*batting_bowling_data.apply(
            lambda row: get_weighted_batting_average(row['batters'].split(', '), balls_faced_list[batting_bowling_data.index.get_loc(row.name)]),
            axis=1
        ))

        batting_bowling_data['weighted_bowling_average'], not_found_bowling = zip(*batting_bowling_data.apply(
            lambda row: get_weighted_bowling_average(row['bowlers'].split(', '), balls_bowled_list[batting_bowling_data.index.get_loc(row.name)]),
            axis=1
        ))

        # Update unique not found players
        for players in not_found_batting:
            unique_not_found_batting.update(players)
        for players in not_found_bowling:
            unique_not_found_bowling.update(players)

        # Save the updated data back to the same file
        batting_bowling_data.to_csv(file_path, index=False)

    # Return summary with lists of not found players
    return {
        "unique_not_found_batting": len(unique_not_found_batting),
        "unique_not_found_bowling": len(unique_not_found_bowling),
        "not_found_batting_players": list(unique_not_found_batting),
        "not_found_bowling_players": list(unique_not_found_bowling)
    }


folder_path = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\innings2\test_processed"
player_stats_path = r"C:\Users\91878\Documents\projects\New folder\odis_male_csv2\modified_file.csv"

result = process_batting_bowling_data(folder_path, player_stats_path)

print(f"Unique players not found for batting: {result['unique_not_found_batting']}")
print(f"Unique players not found for bowling: {result['unique_not_found_bowling']}")
print("Players not found for batting:")
for player in result["not_found_batting_players"]:
    print(player)

print("Players not found for bowling:")
for player in result["not_found_bowling_players"]:
    print(player)
