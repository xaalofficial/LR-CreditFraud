import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc
import joblib

# Load the data
df = pd.read_csv('creditcard.csv')
df = df.dropna()

# Show class imbalance
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (Fraud vs Non-Fraud)')
plt.xlabel('Class (0 = Non-Fraud, 1 = Fraud)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()

# Use a few features for simplicity
X = df[['V1', 'V2', 'V3', 'Amount']]
y = df['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train the model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved as model.pkl")

# Predict on test set
y_pred = model.predict(X_test)

# Print evaluation results
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Plot Precision-Recall Curve
y_scores = model.decision_function(X_test)
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('precision_recall_curve.png')
plt.close()

# Show feature coefficients
coeffs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\n🧠 Feature Coefficients:")
print(coeffs)

# Display sample predictions
sample = X_test.sample(10, random_state=1)
preds = model.predict(sample)
comparison = pd.DataFrame({
    'Actual': y_test.loc[sample.index].values,
    'Predicted': preds
})
print("\n🔍 Sample Predictions:")
print(comparison)
