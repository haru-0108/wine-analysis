import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("../data/wine.csv").to_numpy()

# 説明変数（alcohol）
X = df[:, 10]

# 目的変数（quality）
y = df[:, -1]

# 線形回帰
a, b = np.polyfit(X, y, 1)

# 回帰直線
x_line = np.linspace(X.min(), X.max(), 100)
y_line = a * x_line + b

# グラフ作成
plt.figure(figsize=(6, 5))
plt.plot(X, y, ".", label="Data")
plt.plot(x_line, y_line, "r-", label="Regression Line")

plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Linear Regression")
plt.legend()

# wine-analysis直下に保存
plt.savefig("../linear_regression.png")

plt.show()