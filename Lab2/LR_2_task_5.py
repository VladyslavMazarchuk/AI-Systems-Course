# ===================================================
#   Завдання 2.5 — Класифікація даних Ridge Classifier
# ===================================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# ======== Завантаження набору Iris ========
iris = load_iris()
X, y = iris.data, iris.target

# ======== Поділ на train/test ========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# ======== Класифікатор Ridge ========
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)

# ======== Прогноз ========
y_pred = clf.predict(X_test)

# ======== Метрики ========
print('Accuracy:', np.round(metrics.accuracy_score(y_test, y_pred), 4))
print('Precision:', np.round(metrics.precision_score(y_test, y_pred, average='weighted'), 4))
print('Recall:', np.round(metrics.recall_score(y_test, y_pred, average='weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(y_test, y_pred, average='weighted'), 4))
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(y_test, y_pred), 4))
print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(y_test, y_pred), 4))

print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))

# ======== Матриця плутанини (Confusion Matrix) ========
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title("Confusion Matrix (Ridge Classifier)")
plt.savefig("Confusion.jpg")

# Збереження SVG у буфер
f = BytesIO()
plt.savefig(f, format="svg")

plt.show()
