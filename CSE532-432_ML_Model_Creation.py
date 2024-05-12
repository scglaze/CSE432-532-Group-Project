import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import pickle

df = pd.read_excel("Data/2020to2023_MLB_TeamStats.xlsx")

X = df.drop(columns=["Made Playoffs", "Team", "WHIP", "ERA", "SLG", "BA", "Runs Allowed/Game", "OPS"])
y = df["Made Playoffs"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

models = {
    "Logistic Regression": LogisticRegression(penalty='l2', C=1.0),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=10, min_samples_split=2, min_samples_leaf=1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1),
    "Support Vector Machine": SVC(C=1.0, kernel='rbf', gamma='scale')
}

trained_models = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)
    trained_models[name] = model

combined_predictions = pd.DataFrame(predictions)
combined_predictions["Team"] = df.loc[X_test.index, "Team"].reset_index(drop=True)
combined_predictions["Actual Result"] = y_test.reset_index(drop=True)

num_of_playoff_predictions = combined_predictions.iloc[:, :-3].sum(axis=1)
combined_predictions["Final Prediction"] = num_of_playoff_predictions >= 3
combined_predictions.reset_index(drop=True, inplace=True)

final_accuracy = (combined_predictions["Final Prediction"] == combined_predictions["Actual Result"]).mean()
print("Final Accuracy:", final_accuracy)
combined_predictions.to_excel("combined_predictions.xlsx", index=False)

with open("trained_models.pkl", "wb") as f:
    pickle.dump(trained_models, f)
