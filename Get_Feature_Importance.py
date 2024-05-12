import json
import pickle
import pandas as pd

df = pd.read_excel("Data/2020to2023_MLB_TeamStats.xlsx")
df = df.drop(columns=["Made Playoffs", "Team", "WHIP", "ERA", "SLG", "BA", "Runs Allowed/Game", "OPS"])

with open("trained_models.pkl", "rb") as f:
    trained_models = pickle.load(f)

feature_names = df.columns.tolist()

feature_importance_data = {}

for name, model in trained_models.items():
    if name in ["Gradient Boosting", "Decision Tree", "Random Forest"]:
        feature_importances = model.feature_importances_
        feature_importance_dict = {feature_names[i]: round(importance, 4) for i, importance in enumerate(feature_importances)}
        feature_importance_data[name] = feature_importance_dict
    elif name == "Logistic Regression": 
        coefficients = model.coef_[0]
        feature_importance_dict = {feature_names[i]: round(coefficient, 4) for i, coefficient in enumerate(coefficients)}
        feature_importance_data[name] = feature_importance_dict
    elif name == "Support Vector Machine":
        pass
    else:
        print("Incorrect name for ", name)

with open("feature_importance_data.json", "w") as json_file:
    json.dump(feature_importance_data, json_file, indent=4)

print("Feature importance data written to feature_importance_data.json")