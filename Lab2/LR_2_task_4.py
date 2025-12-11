import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ===========================
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ===========================
input_file = "income_data.txt"

X_raw = []
y_raw = []

limit = 25000
c1 = c2 = 0

with open(input_file, "r") as f:
    for line in f:
        if c1 >= limit and c2 >= limit:
            break
        if "?" in line:
            continue  # пропускаємо неповні дані

        row = line.strip().split(", ")
        label = row[-1]

        if label == "<=50K" and c1 < limit:
            X_raw.append(row[:-1])
            y_raw.append(label)
            c1 += 1
        elif label == ">50K" and c2 < limit:
            X_raw.append(row[:-1])
            y_raw.append(label)
            c2 += 1

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

# ===========================
# 2. КОДУВАННЯ ОЗНАК
# ===========================
X_encoded = np.zeros(X_raw.shape)
encoders = []

for col in range(X_raw.shape[1]):
    if X_raw[0, col].replace(".", "", 1).isdigit():
        X_encoded[:, col] = X_raw[:, col].astype(float)
        encoders.append(None)
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, col] = le.fit_transform(X_raw[:, col])
        encoders.append(le)

# Кодування цілі
label_enc = preprocessing.LabelEncoder()
y_encoded = label_enc.fit_transform(y_raw)

# ===========================
# 3. МАСШТАБУВАННЯ
# ===========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# ===========================
# 4. РОЗБИТТЯ ДАНИХ
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=5, stratify=y_encoded
)

# ===========================
# 5. СПИСОК МОДЕЛЕЙ
# ===========================
models = [
    ("LR", LogisticRegression(max_iter=2000)),
    ("LDA", LinearDiscriminantAnalysis()),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier()),
    ("NB", GaussianNB()),
    ("SVM", SVC())
]

print("\n=== Результати 10-fold Cross Validation ===")
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_scores)
    names.append(name)
    print(f"{name}: {cv_scores.mean():.4f} ({cv_scores.std():.4f})")

# ===========================
# 6. ГРАФІК ПОРІВНЯННЯ
# ===========================
plt.boxplot(results, labels=names)
plt.title("Порівняння алгоритмів класифікації (Income Dataset)")
plt.ylabel("Accuracy")
plt.grid(alpha=0.3)
plt.show()

# ===========================
# 7. ФІНАЛЬНА ОЦІНКА ЛУЧШОЇ МОДЕЛІ
# ===========================
best_model = SVC()
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)

print("\n=== Оцінка найкращої моделі на тестовому наборі ===")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("Classification report:\n", classification_report(y_test, pred))
