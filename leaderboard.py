import json
from typing import Dict, Any
import time
from datetime import datetime
import threading
import config
import math
import os
from bt_mle import compute_bt_scores

# define local file path
LEADERBOARD_FILE = "leaderboard.json"
BACKUP_FOLDER = "leaderboard_backups"
# ensure backup directory exists
if not os.path.exists(BACKUP_FOLDER):
    os.makedirs(BACKUP_FOLDER)

# Dictionary to store ELO ratings
elo_ratings = {}

def load_leaderboard() -> Dict[str, Any]:
    try:
        if os.path.exists(LEADERBOARD_FILE):
            with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}  # 如果文件不存在，返回空字典
    except Exception as e:
        print(f"Error loading leaderboard: {str(e)}")
        return {}

def save_leaderboard(leaderboard_data: Dict[str, Any]) -> bool:
    try:
        with open(LEADERBOARD_FILE, 'w', encoding='utf-8') as f:
            json.dump(leaderboard_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving leaderboard: {str(e)}")
        return False

def get_model_size(model_name):
    """Extract model size in billions from model name.
    Handles various formats like:
    - "Model 14B (4-bit)"
    - "Model (14B)"
    - "Model 14.5B"
    - "Model 1,000M"
    """
    for model, human_readable in config.get_approved_models():
        if model == model_name:
            try:
                # Remove any commas
                clean_name = human_readable.replace(',', '')
                
                # Try to find size in parentheses first
                if '(' in clean_name:
                    parts = clean_name.split('(')
                    for part in parts:
                        if 'B' in part:
                            size_str = part.split('B')[0].strip()
                            try:
                                return float(size_str)
                            except ValueError:
                                continue
                
                # If not in parentheses, look for B or M in the whole string
                words = clean_name.split()
                for word in words:
                    if 'B' in word:
                        size_str = word.replace('B', '').strip()
                        try:
                            return float(size_str)
                        except ValueError:
                            continue
                    elif 'M' in word:
                        size_str = word.replace('M', '').strip()
                        try:
                            return float(size_str) / 1000  # Convert millions to billions
                        except ValueError:
                            continue
                            
            except Exception as e:
                print(f"Error parsing size for {model_name}: {e}")
                
    return 1.0  # Default size if not found or parsing failed

def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

def update_elo_ratings(winner, loser, tie=False):
    if winner not in elo_ratings or loser not in elo_ratings:
        initialize_elo_ratings()
    
    winner_rating = elo_ratings[winner]
    loser_rating = elo_ratings[loser]
    
    expected_winner = calculate_expected_score(winner_rating, loser_rating)
    expected_loser = 1 - expected_winner
    
    # winner_size = get_model_size(winner)
    # loser_size = get_model_size(loser)
    # max_size = max(get_model_size(model) for model, _ in config.get_approved_models())
    
    # k_factor = min(64, 32 * (1 + (loser_size - winner_size) / max_size))
    k_factor = 4 # refer to GenAI-Arena paper for ELO calculation
    if not tie:
        elo_ratings[winner] += k_factor * (1 - expected_winner)
        elo_ratings[loser] += k_factor * (0 - expected_loser)
    else:
        elo_ratings[winner] += k_factor * (0.5 - expected_winner)
        elo_ratings[loser] += k_factor * (0.5 - expected_loser)

def initialize_elo_ratings():
    leaderboard = load_leaderboard()
    for model, _ in config.get_approved_models():
        size = get_model_size(model)
        # elo_ratings[model] = 1000 + (size * 100)
        elo_ratings[model] = 1000
    
    # Replay all battles to update ELO ratings
    for model, data in leaderboard.items():
        if model not in elo_ratings:
            # elo_ratings[model] = 1000 + (get_model_size(model) * 100)
            elo_ratings[model] = 1000
        for opponent, results in data['opponents'].items():
            if opponent not in elo_ratings:
                # elo_ratings[opponent] = 1000 + (get_model_size(opponent) * 100)
                elo_ratings[opponent] = 1000
            for _ in range(results['wins']):
                update_elo_ratings(model, opponent)
            for _ in range(results['losses']):
                update_elo_ratings(opponent, model)
            for _ in range(results['ties']):
                update_elo_ratings(model, opponent, tie=True)

def ensure_elo_ratings_initialized():
    if not elo_ratings:
        initialize_elo_ratings()

def update_leaderboard(winner: str, loser: str, image_path: str, text_prompt: str, winner_response: str, loser_response: str, tie: bool) -> Dict[str, Any]:
    leaderboard = load_leaderboard()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    if winner not in leaderboard:
        leaderboard[winner] = {"wins": 0, "losses": 0, "ties": 0, "opponents": {}, "history": []}
    if loser not in leaderboard:
        leaderboard[loser] = {"wins": 0, "losses": 0, "ties": 0, "opponents": {}, "history": []}
    
    # ensure there are images attribute
    if "history" not in leaderboard[winner]:
        leaderboard[winner]["history"] = []
    if "history" not in leaderboard[loser]:
        leaderboard[loser]["history"] = []

    if not tie:
        leaderboard[winner]["history"].append({
            "result": "win",
            "opponent": loser,
            "text_prompt": text_prompt,
            "image": image_path,
            "timestamp": now,
            "response": winner_response,
            "opponent_response": loser_response,
        })
        leaderboard[loser]["history"].append({
            "result": "loss",
            "opponent": winner,
            "text_prompt": text_prompt,
            "image": image_path,
            "timestamp": now,
            "response": loser_response,
            "opponent_response": winner_response,
        })
        leaderboard[winner]["wins"] += 1
        leaderboard[winner]["opponents"].setdefault(loser, {"wins": 0, "losses": 0, "ties": 0})["wins"] += 1
        
        leaderboard[loser]["losses"] += 1
        leaderboard[loser]["opponents"].setdefault(winner, {"wins": 0, "losses": 0, "ties": 0})["losses"] += 1
        
        # Update ELO ratings
        update_elo_ratings(winner, loser)
        
        save_leaderboard(leaderboard)
        return leaderboard
    else:
        leaderboard[winner]["history"].append({
            "result": "tie",
            "opponent": loser,
            "text_prompt": text_prompt,
            "image": image_path,
            "timestamp": now,
            "response": winner_response,
            "opponent_response": loser_response,
        })
        leaderboard[loser]["history"].append({
            "result": "tie",
            "opponent": winner,
            "text_prompt": text_prompt,
            "image": image_path,
            "timestamp": now,
            "response": loser_response,
            "opponent_response": winner_response,
        })
        leaderboard[winner]["ties"] += 1
        leaderboard[winner]["opponents"].setdefault(loser, {"wins": 0, "losses": 0, "ties": 0})["ties"] += 1
        leaderboard[loser]["ties"] += 1
        leaderboard[loser]["opponents"].setdefault(winner, {"wins": 0, "losses": 0, "ties": 0})["ties"] += 1

        # Update ELO ratings
        update_elo_ratings(winner, loser, tie=True)
        save_leaderboard(leaderboard)
        return leaderboard
    

def get_current_leaderboard() -> Dict[str, Any]:
    return load_leaderboard()

def get_human_readable_name(model_name: str) -> str:
    model_dict = dict(config.get_approved_models())
    return model_dict.get(model_name, model_name)

def get_leaderboard():
    leaderboard = load_leaderboard()

    all_history = []
    for model, entry in leaderboard.items():
        if 'history' not in entry:
            continue
        for h in entry['history']:
            if h['result'] == "win":
                all_history.append({'model': model, 'opponent': h['opponent'], 'result': "win"})
            elif h['result'] == "loss":
                # 忽略，另一方会记录为 win，避免重复
                continue
            elif h['result'] == "tie":
                # 只记一次
                if model < h['opponent']:
                    all_history.append({'model': model, 'opponent': h['opponent'], 'result': "tie"})
    # bt_scores = compute_bt_scores(all_history)
    bt_scores = compute_bt_scores(leaderboard)
    
    # Prepare data for Gradio table
    table_data = []
    headers = ["#", "Model", "BT Score", "Wins", "Losses", "Ties", "Total Battles", "Win Rate"]
    # all_models = set(dict(config.get_approved_models()).keys()) | set(leaderboard.keys())
    all_models = set(dict(config.get_approved_models()).keys()) # just show approved models
    
    # for model, results in leaderboard.items():
    for model in all_models:
        results = leaderboard.get(model, {'wins': 0, 'losses': 0, 'ties': 0})
        wins = results.get('wins', 0)
        losses = results.get('losses', 0)
        ties = results.get('ties', 0)
        total_battles = wins + losses + ties
        
        # Calculate win rate
        win_rate = wins / total_battles if total_battles > 0 else 0
        
        # Calculate score using the formula: win_rate * (1 - 1/(total_battles + 1))
        # score = win_rate * (1 - 1/(total_battles + 1)) if total_battles > 0 else 0
        bt_score = bt_scores.get(model, 1000)
        
        # Get human readable name
        human_readable = get_human_readable_name(model)
        
        # Format the row with formatted strings for display
        row = [
            0,                    # Position placeholder (integer)
            human_readable,       # String
            f"{bt_score:.1f}",      # Score formatted to 3 decimal places
            wins,                 # Integer
            losses,              # Integer
            ties,                # Integer
            total_battles,       # Integer
            f"{win_rate:.1%}"    # Win rate as percentage
        ]
        table_data.append(row)
    
    # Sort by score (descending)
    table_data.sort(key=lambda x: float(x[2].replace('%', '')), reverse=True)
    
    # Add position numbers after sorting
    for i, row in enumerate(table_data, 1):
        row[0] = i
    
    return table_data

# def calculate_elo_impact(model):
#     positive_impact = 0
#     negative_impact = 0
#     leaderboard = load_leaderboard()
#     initial_rating = 1000 + (get_model_size(model) * 100)
    
#     if model in leaderboard:
#         for opponent, results in leaderboard[model]['opponents'].items():
#             model_size = get_model_size(model)
#             opponent_size = get_model_size(opponent)
#             max_size = max(get_model_size(m) for m, _ in config.get_approved_models())
            
#             size_difference = (opponent_size - model_size) / max_size
            
#             win_impact = 1 + max(0, size_difference)
#             loss_impact = 1 + max(0, -size_difference)
            
#             positive_impact += results['wins'] * win_impact
#             negative_impact += results['losses'] * loss_impact
    
#     return round(positive_impact), round(negative_impact), round(initial_rating)

def get_elo_leaderboard():
    ensure_elo_ratings_initialized()
    
    # Prepare data for Gradio table
    table_data = []
    headers = ["#", "Model", "ELO Rating", "Wins", "Losses", "Ties", "Total Battles", "Win Rate"]
    
    leaderboard = load_leaderboard()
    # all_models = set(dict(config.get_approved_models()).keys()) | set(leaderboard.keys())
    all_models = set(dict(config.get_approved_models()).keys()) # just show approved models
    
    for model in all_models:
        # Get ELO rating
        # rating = elo_ratings.get(model, 1000 + (get_model_size(model) * 100))
        rating = elo_ratings.get(model, 1000)
        
        # Get battle data
        wins = leaderboard.get(model, {}).get('wins', 0)
        losses = leaderboard.get(model, {}).get('losses', 0)
        ties = leaderboard.get(model, {}).get('ties', 0)
        total_battles = wins + losses + ties
        win_rate = wins / total_battles if total_battles > 0 else 0
        
        # Get human readable name
        human_readable = get_human_readable_name(model)
        
        # Format the row with formatted strings for display
        row = [
            0,                    # Position placeholder (integer)
            human_readable,       # String
            f"{rating:.1f}",     # ELO rating formatted to 1 decimal place
            wins,                 # Integer
            losses,              # Integer
            ties,                # Integer
            total_battles,       # Integer
            f"{win_rate:.1%}"    # Win rate as percentage
        ]
        table_data.append(row)
    
    # Sort by ELO rating (descending)
    table_data.sort(key=lambda x: float(x[2]), reverse=True)
    
    # Add position numbers after sorting
    for i, row in enumerate(table_data, 1):
        row[0] = i
    
    return table_data

# def create_backup():
#     while True:
#         try:
#             leaderboard_data = load_leaderboard()
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             backup_file_name = f"leaderboard_backup_{timestamp}.json"
#             backup_path = f"{config.NEXTCLOUD_BACKUP_FOLDER}/{backup_file_name}"
#             json_data = json.dumps(leaderboard_data, indent=2)
#             nc.files.upload(backup_path, json_data.encode('utf-8'))
#             print(f"Backup created on Nextcloud: {backup_path}")
#         except Exception as e:
#             print(f"Error creating backup: {e}")
#         time.sleep(43200)  # Sleep for 12 HOURS
def create_backup():
    while True:
        try:
            leaderboard_data = load_leaderboard()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file_name = f"leaderboard_backup_{timestamp}.json"
            backup_path = os.path.join(BACKUP_FOLDER, backup_file_name)
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(leaderboard_data, f, indent=2)
            print(f"Backup created locally: {backup_path}")
        except Exception as e:
            print(f"Error creating backup: {e}")
        time.sleep(86400)  # Sleep for 24 HOURS

def start_backup_thread():
    backup_thread = threading.Thread(target=create_backup, daemon=True)
    backup_thread.start()