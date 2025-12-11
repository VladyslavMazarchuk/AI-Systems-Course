import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

input_file = "income_data.txt"

X_raw = []
y_raw = []

limit = 25000
count1 = count2 = 0

with open(input_file, "r") as f:
    for line in f:
        if count1 >= limit and count2 >= limit:
            break
        if "?" in line:
            continue

        row = line.strip().split(", ")
        label = row[-1]

        if label == "<=50K" and count1 < limit:
            X_raw.append(row[:-1])
            y_raw.append(label)
            count1 += 1
        elif label == ">50K" and count2 < limit:
            X_raw.append(row[:-1])
            y_raw.append(label)
            count2 += 1

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

encoders = []
X_encoded = np.zeros(X_raw.shape)

for col in range(X_raw.shape[1]):
    if X_raw[0, col].replace(".", "", 1).isdigit():
        X_encoded[:, col] = X_raw[:, col].astype(float)
        encoders.append(None)
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, col] = le.fit_transform(X_raw[:, col])
        encoders.append(le)

label_enc = preprocessing.LabelEncoder()
y_encoded = label_enc.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=5
)

# ========= RBF ЯДРО =========
model = SVC(kernel="rbf", gamma="scale")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== SVM з гаусовим ядром (RBF) ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
