import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# -------------------------------
# Генерація даних (ВАРІАНТ 9)
# -------------------------------
np.random.seed(42)
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 3 + np.sin(X) + np.random.uniform(-0.5, 0.5, (m, 1))

# -------------------------------
# ЛІНІЙНА РЕГРЕСІЯ
# -------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# -------------------------------
# ПОЛІНОМІАЛЬНА РЕГРЕСІЯ (2 ступінь)
# -------------------------------
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

y_poly_pred = poly_reg.predict(X_poly)

# -------------------------------
# ОЦІНКА ЯКОСТІ
# -------------------------------
print("=== Лінійна регресія ===")
print(f"R2 score: {r2_score(y, y_lin_pred):.4f}")
print(f"Рівняння: y = {lin_reg.coef_[0][0]:.3f}x + ({lin_reg.intercept_[0]:.3f})")

print("\n=== Поліноміальна регресія (2-й ступінь) ===")
print(f"R2 score: {r2_score(y, y_poly_pred):.4f}")
print(f"Коефіцієнти: {poly_reg.coef_[0]}")
print(f"Рівняння: y = {poly_reg.coef_[0][1]:.3f}x² + "
      f"{poly_reg.coef_[0][0]:.3f}x + ({poly_reg.intercept_[0]:.3f})")

# -------------------------------
# ПОБУДОВА ГРАФІКА
# -------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Дані (3 + sin(x) + шум)')
plt.plot(X, y_lin_pred, color='red', linestyle='--', linewidth=2, label='Лінійна регресія')
plt.plot(X, y_poly_pred, color='green', linewidth=3, label='Поліноміальна регресія (deg=2)')

plt.title('Завдання 2.5 — Самостійна побудова регресії (Варіант 9)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
