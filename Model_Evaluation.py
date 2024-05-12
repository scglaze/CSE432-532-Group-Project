import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_excel("combined_predictions.xlsx")

accuracy = accuracy_score(data["Actual Result"], data["Final Prediction"])

precision = precision_score(data["Actual Result"], data["Final Prediction"])

recall = recall_score(data["Actual Result"], data["Final Prediction"])

f1 = f1_score(data["Actual Result"], data["Final Prediction"])

conf_matrix = confusion_matrix(data["Actual Result"], data["Final Prediction"])

plt.figure(figsize=(10, 10))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Predicted FALSE', 'Predicted TRUE'], rotation=45)
plt.yticks([0, 1], ['Actual FALSE', 'Actual TRUE'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
plt.savefig('confusion_matrix.png')

fpr, tpr, thresholds = roc_curve(data["Actual Result"], data["Final Prediction"])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", roc_auc)