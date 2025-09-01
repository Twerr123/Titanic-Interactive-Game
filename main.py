# Tyler Werr

# This Titanic interactive game uses real statistical data to predict the user's.
# The training machine learning model specifically the Random Forest model learns about
# survival patterns based on the input features listed below.
# Steps to run game via terminal
# python3 train_model.py (or whatever version of python you have)
# python3 main.py
# logs data into a csv file after each round

import joblib
import pandas as pd
from logger import log_game

model = joblib.load('model/titanic_model.pkl')

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] # list of strings for features variable

def main():
    player_data = {} # the dict stores the answer of the player input

    for feature in features: # for loop is a way to repeat an action for each item
        if feature == 'Pclass':
            value = input("Choose your passenger class (1 = 1st class, 2 = 2nd class, 3 = 3rd class): ")
        elif feature == 'Sex':
            value = input("Choose your sex (male/female, 0 or 1): ")
            value = 0 if value.lower() == 'male' else 1
        elif feature == 'Age':
            value = input("Enter your age (number): ")
        elif feature == 'SibSp':
            value = input("How many siblings/spouses are traveling with you? (number): ")
        elif feature == 'Parch':
            value = input("How many parents/children are traveling with you? (number): ")
        elif feature == 'Fare':
            value = input("Enter your ticket fare (approximate number, like 7.25 or 50.0): ")
        elif feature == 'Embarked':
            value = input("Choose port which you embarked from (C = Cherbourg, Q = Queenstown, S = Southampton): ")
            value = {'C': 0, 'Q': 1, 'S': 2}.get(value.upper(), 2)  # Map C, Q, S to 0, 1, 2

        # this block is necessary because machine learning expects numbers not strings or characters
        if feature in ['Fare', 'Age']:
            player_data[feature] = float(value)
        else:
            player_data[feature] = int(value) # for all other features convert to int

    player_df = pd.DataFrame([player_data], columns=features) # ensures a list with players answers in column format

    prediction = model.predict(player_df)[0] # given these inputs predict if they survived
    probabilities = model.predict_proba(player_df)[0] # returns list of probabilities
    survival_chance = probabilities[1] # Retrieves survival probability index 1 only
    percent = survival_chance * 100
    if prediction == 1:
      if percent >= 90:
          print(f"You survived! (Survival odds: {percent:.1f}%) You easily made it off the ship!")
      elif percent >= 70:
          print(f"You survived! (Survival odds: {percent: .1f}% You likely survived the catastrophy!")
      elif percent >= 50:
          print(f"You survived! (Survival odds: {percent: .1f}% You barely made it out alive..")
      else:
        print(f"You survived! {percent: .1f} You were lucky to get out alive..") # in the case of < 50
    else:
        print(f"You did not survive. (Survival odds: {percent: .1f}")

    return player_data, prediction, percent

# start the game
if __name__ == "__main__":
    while True:
        player_data, prediction, percent = main()
        log_game(player_data, prediction, percent)
        play_again = input("Do you want to play again (yes/no): ").lower()
        if play_again != 'yes': # if not equal to yes
            print("Exiting..")
            break














