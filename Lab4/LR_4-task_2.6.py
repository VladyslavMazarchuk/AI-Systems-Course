import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ================================
# Генерація даних (як у завданні 2.5)
# ================================
np.random.seed(42)
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 3 + np.sin(X) + np.random.uniform(-0.5, 0.5, (m, 1))

# ================================
# Функція побудови кривих навчання
# ================================
def plot_learning_curves(model, X, y, title, ax):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    train_errors = []
    val_errors = []

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])

        y_train_pred = model.predict(X_train[:m])
        y_val_pred = model.predict(X_val)

        train_errors.append(
            np.sqrt(mean_squared_error(y_train[:m], y_train_pred))
        )
        val_errors.append(
            np.sqrt(mean_squared_error(y_val, y_val_pred))
        )

    ax.plot(train_errors, "r-+", linewidth=2, label="Навчальний набір")
    ax.plot(val_errors, "b-", linewidth=3, label="Перевірочний набір")
    ax.set_title(title)
    ax.set_xlabel("Розмір навчального набору")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True)

# ================================
# Побудова графіків
# ================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) Лінійна регресія (недонавчання)
linear_reg = LinearRegression()
plot_learning_curves(
    linear_reg,
    X,
    y,
    "Лінійна регресія (Недонавчання)",
    axes[0]
)

# 2) Поліноміальна регресія 10-го ступеня (перенавчання)
poly_reg_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression())
])

plot_learning_curves(
    poly_reg_10,
    X,
    y,
    "Поліноміальна регресія 10-го ступеня (Перенавчання)",
    axes[1]
)

# 3) Поліноміальна регресія 2-го ступеня (оптимальна модель)
poly_reg_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression())
])

plot_learning_curves(
    poly_reg_2,
    X,
    y,
    "Поліноміальна регресія 2-го ступеня (Оптимальна)",
    axes[2]
)

plt.tight_layout()
plt.show()
