import pandas as pd # pandas is used to load, clean, manipulate data like csv files
from sklearn.ensemble import RandomForestClassifier # machine learning algorithm used to predict Titanic survival
from sklearn.model_selection import train_test_split # splits datasets into training and testing sets
from sklearn.metrics import accuracy_score # calculates accuracy of model's predictions seeing how well model performed on test data
import joblib # so that I can save trained model and reuse it in game without retraining
import os

df = pd.read_csv('data/train.csv') # reads a csv file and loads date into dataframe

df['Age'].fillna(df['Age'].median(), inplace=True) # age column in dataframe. fills missing values. calculates middle value for ages
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) # column of which port passenger arrived from. fills missing values. mode gets most fequent value

# numerically encoded columns
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}) # sex column in dataframe. genders labeled 0 or 1
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}) # maps boarding ports with numbers

# SibSp = siblings/spouses, Parch = parents/children
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] # creates list of column names to use as features for the model
X = df[features] # X variable notifies the model on the information to use to make predictions. input
y = df['Survived'] # this is what the model will predict. output

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50) # these numbers say 20% will go into testing, and random_state says that the same people will go into training and testing each time

# First line sets up model. Second trains model with fits(train)
model = RandomForestClassifier(n_estimators=100, random_state=50) # Random Forest is an ML algorithm that builds decision trees and averages results to make prediction. 100 decision trees
model.fit(X_train, y_train) # X is for the features, y is for target outcomes

y_pred = model.predict(X_test) # uses trained Random Forest model to predict survival of passengers from X test set. y_pred stores predicted results (0 or 1)
accuracy = accuracy_score(y_test, y_pred) # compares true survival labels with models prediction to get an accuracy percentage
print(f"Model accuracy: {accuracy: .2%}") # prints accuracy as percentage with two decimal places


os.makedirs("model", exist_ok=True) # makes folder to save file to
joblib.dump(model, 'model/titanic_model.pkl') # saves trained model into file
print("Model saved to model/titanic_model.pkl")













