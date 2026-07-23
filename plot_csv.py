# %%
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the CSV file
df = pd.read_csv('RESNET18_CIFAR10_dense_seed0.csv')
# print(df["accuracy"])
data = df["accuracy"]
plt.plot(data)

# 2. Plot all numeric columns instantly
# df.plot()
plt.savefig('plot.png')

# 3. Display the graph
plt.show()

# %%
