import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === Завантаження та попередня обробка даних ===

dataset_path = "income_data.txt"

raw_features = []
raw_labels = []
limit_per_class = 25000
c1, c2 = 0, 0

with open(dataset_path, "r") as file:
    for row in file:
        if c1 >= limit_per_class and c2 >= limit_per_class:
            break
        if '?' in row:
            continue

        parts = row.strip().split(", ")

        label = parts[-1]

        # Розподіл на два класи
        if label == "<=50K" and c1 < limit_per_class:
            raw_features.append(parts)
            c1 += 1
        elif label == ">50K" and c2 < limit_per_class:
            raw_features.append(parts)
            c2 += 1

# Перетворення у numpy-масив
raw_features = np.array(raw_features)

# === Кодування категоріальних ознак ===

encoders = []                     # зберігатимемо всі енкодери
encoded_matrix = np.zeros(raw_features.shape)

for col_idx, example_value in enumerate(raw_features[0]):
    # Перевірка чи значення є числовим
    if example_value.replace('.', '', 1).isdigit():
        encoded_matrix[:, col_idx] = raw_features[:, col_idx].astype(float)
    else:
        encoder = preprocessing.LabelEncoder()
        encoded_matrix[:, col_idx] = encoder.fit_transform(raw_features[:, col_idx])
        encoders.append(encoder)

# Поділ на X та y
X_data = encoded_matrix[:, :-1].astype(float)
y_data = encoded_matrix[:, -1].astype(int)

# === Розбиття вибірки ===

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

# === Побудова SVM-класифікатора ===

svm_clf = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=7000))
svm_clf.fit(X_train, y_train)

# === Прогнозування ===

y_pred = svm_clf.predict(X_test)

# === Обчислення метрик ===

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("=== Оцінка якості класифікації ===")
print(f"Точність (Accuracy): {acc:.4f}")
print(f"Точність (Precision): {prec:.4f}")
print(f"Повнота (Recall): {rec:.4f}")
print(f"F1-міра: {f1:.4f}")

# === Передбачення для одного прикладу ===

test_person = [
    '37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
    'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States'
]

encoded_test = []      # закодована точка
enc_idx = 0            # індекс енкодера для категоріальних значень

for attr in test_person:
    if attr.replace('.', '', 1).isdigit():
        encoded_test.append(float(attr))
    else:
        encoded_test.append(encoders[enc_idx].transform([attr])[0])
        enc_idx += 1

encoded_test = np.array(encoded_test).reshape(1, -1)

pred_class = svm_clf.predict(encoded_test)
final_label = encoders[-1].inverse_transform(pred_class)[0]

print("\n=== Результат класифікації тестової точки ===")
print(f"Тестова точка належить до класу: {final_label}")
