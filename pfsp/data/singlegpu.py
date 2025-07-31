# Preliminary data treatment file
import pandas as pd

df = pd.read_csv("gpu.csv")

# Example: average total time per instance
print(df.groupby("instance_id")["total_time"].mean())

# Example: boxplot
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=df, x="instance_id", y="total_time")
plt.show()
