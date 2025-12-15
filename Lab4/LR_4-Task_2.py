import numpy as np
import matplotlib.pyplot as plt

# дані 
x = np.array([2, 4, 6, 8, 10, 12])
y = np.array([6.5, 4.4, 3.8, 3.5, 3.1, 3.0])

# Метод найменших квадратів (лінійна апроксимація)
a, b = np.polyfit(x, y, 1)

# Обчислення апроксимованих значень
y_pred = a * x + b

# Побудова графіка
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', label='Експериментальні точки', zorder=5)
plt.plot(x, y_pred, color='blue', label=f'Апроксимація: y = {a:.3f}x + {b:.3f}')

plt.title('Апроксимація методом найменших квадратів')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--')
plt.legend()
plt.show()