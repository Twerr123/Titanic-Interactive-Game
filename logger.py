import pandas as pd
from datetime import datetime
import os

def log_game(player_data, prediction, percent):
    timestamp = datetime.now().strftime("%Y-%m-%d %H: %M: %S") # represents year, month, day, hour, minute, second (e.g. 2025-04-29 14:35:09)

    # Creates DataFrame row
    log_row = pd.DataFrame([{
        'Timestamp': timestamp,
        **player_data, # Unpacks player data keys: Pclass, Sex, etc.
        'Prediction': prediction,
        'Survival Chance': round(percent, 2)
    }])

    if os.path.exists("game_log.csv"): # check if game_log.csv already exists
        log_row.to_csv("game_log.csv", mode='a', header=False, index=False) # appends new data to end of file without rewriting column headers
    else:
        log_row.to_csv("game_log.csv", mode='w', header=True, index=False) # write starts a new file including headers, does not include extra index columns