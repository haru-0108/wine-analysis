import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# データ読み込み
df = pd.read_csv("winequality-red.csv")

# 説明変数
X = df[["alcohol"]]

# 目的変数
y = df["quality"]

# 学習
model = LinearRegression()
model.fit(X, y)

# 予測
pred = model.predict(X)

# グラフ
plt.scatter(X, y)
plt.plot(X, pred, color="red")
plt.xlabel("Alcohol")
plt.ylabel("Quality")
plt.title("Linear Regression")

# 保存
plt.savefig("linear_regression.png")
plt.show()