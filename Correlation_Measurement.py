import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


df = pd.read_excel("Data/2020to2023_MLB_TeamStats.xlsx")
df = df.drop(columns=["Made Playoffs", "Team", "WHIP", "ERA", "SLG", "BA", "Runs Allowed/Game", "OPS"])

correlation_matrix = df.corr()

high_correlation_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            feature_i = correlation_matrix.columns[i]
            feature_j = correlation_matrix.columns[j]
            high_correlation_features.add((feature_i, feature_j))


feature_counts = Counter()
for pair in high_correlation_features:
    feature_counts.update(pair)

sorted_feature_counts = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

print("Sorted features by occurrence count in high_correlation_features:")
for feature, count in sorted_feature_counts:
    print(feature, count)


features_with_high_correlation = set()

for pair in high_correlation_features:
    feature1, feature2 = pair
    correlation = correlation_matrix.loc[feature1, feature2]
    if correlation > 0.9:
        features_with_high_correlation.add(feature1)
        features_with_high_correlation.add(feature2)

print("Features involved in pairs with correlation > 0.9:")
print(features_with_high_correlation)

# Plotting the heatmap
#plt.figure(figsize=(15, 15))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
#plt.title('Correlation Heatmap of Features')
#plt.savefig('correlation_heatmap.png')
#plt.show()