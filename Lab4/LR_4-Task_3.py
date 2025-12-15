import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані
x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

# 1. Матриця Вандермонда
X_matrix = np.vander(x, 5)
print("Матриця X (Вандермонда):")
print(X_matrix)
print("-" * 30)

# 2. Знаходження коефіцієнтів полінома 4-го степеня
degree = 4
coefficients = np.polyfit(x, y, degree)

print("Коефіцієнти інтерполяційного полінома (a4 ... a0):")
print(coefficients)
print("-" * 30)

# 3. Поліном
poly_func = np.poly1d(coefficients)
print("Інтерполяційний поліном:")
print(poly_func)
print("-" * 30)

# 4. Значення в проміжних точках
x_targets = [0.2, 0.5]
print("Значення функції в проміжних точках:")
for val in x_targets:
    print(f"x = {val} -> y = {poly_func(val):.4f}")

# 5. Побудова графіка
x_smooth = np.linspace(min(x), max(x), 100)
y_smooth = poly_func(x_smooth)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', s=80, label='Задані точки', zorder=5)
plt.plot(x_smooth, y_smooth, color='blue', label='Інтерполяційний поліном (4-го степеня)')
plt.scatter(x_targets, poly_func(x_targets), color='green', marker='x',
            s=100, label='Значення в точках 0.2 та 0.5', zorder=6)

plt.title('Інтерполяція поліномом 4-го степеня')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--')
plt.legend()
plt.show()
