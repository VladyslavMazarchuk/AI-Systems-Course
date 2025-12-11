# ==============================
#   КРОК 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ==============================

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

# Завантаження датасету
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Перевірки
print("\n=== Ключова інформація ===")
print("Розмір датасету:", dataset.shape)
print("\nПерші 20 рядків:")
print(dataset.head(20))
print("\nОписова статистика:")
print(dataset.describe())
print("\nКількість прикладів кожного класу:")
print(dataset.groupby('class').size())

# ==============================
#   КРОК 2. ВІЗУАЛІЗАЦІЯ ДАНИХ
# ==============================

print("\n=== Побудова графіків... ===")

# діаграми розмаху
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# гістограми
dataset.hist()
pyplot.show()

# матриця діаграм розсіювання
scatter_matrix(dataset)
pyplot.show()

# ==============================
#   КРОК 3. РОЗДІЛЕННЯ ДАНИХ
# ==============================

array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=0.20, random_state=1)

# ==============================
#   КРОК 4. ПОБУДОВА МОДЕЛЕЙ
# ==============================

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

print("\n=== Результати 10-кратної крос-валідації ===")

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# діаграма розмаху точностей
pyplot.boxplot(results, labels=names)
pyplot.title('Порівняння алгоритмів')
pyplot.show()

# ==============================
#   КРОК 5. ОЦІНКА НА ТЕСТОВОМУ НАБОРІ
# ==============================

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\n=== Оцінка якості на тестовому наборі ===")
print("Accuracy:", accuracy_score(Y_validation, predictions))
print("\nМатриця помилок:\n", confusion_matrix(Y_validation, predictions))
print("\nЗвіт класифікації:\n", classification_report(Y_validation, predictions))

# ==============================
#   КРОК 6. ПРОГНОЗ ДЛЯ НОВОЇ КВІТКИ
# ==============================

X_new = np.array([[5, 2.9, 1, 0.2]])
print("\nФорма нового прикладу:", X_new.shape)

new_prediction = model.predict(X_new)
print("\nПрогноз для нової квітки:", new_prediction[0])
