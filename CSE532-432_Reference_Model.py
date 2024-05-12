import pandas as pd
import pickle

df = pd.read_excel("Data/2020to2023_MLB_TeamStats.xlsx")
df = df.drop(columns=["Made Playoffs", "Team", "WHIP", "ERA", "SLG", "BA", "Runs Allowed/Game", "OPS"])

with open("trained_models.pkl", "rb") as f:
    trained_models = pickle.load(f)

data = pd.read_excel("2024_MLB_TeamStats.xlsx")

X = data.drop(columns=["Made Playoffs", "Team"])

predictions = {}
for name, model in trained_models.items():
    predictions[name] = model.predict(X)

combined_predictions = pd.DataFrame(predictions)
combined_predictions["Team"] = data["Team"]
combined_predictions["Count"] = combined_predictions.iloc[:, :5].sum(axis=1)
combined_predictions["Final Prediction"] = combined_predictions["Count"] >= 3

combined_predictions.to_excel("2024_Predictions.xlsx", index=False)