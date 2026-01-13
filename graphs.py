import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("tilt_data.csv")

# Basic info
print(df.head())
print(df.describe())

# 1. Session length vs tilt
plt.figure()
plt.scatter(df["session_seconds"], df["tilted"])
plt.xlabel("Session Seconds")
plt.ylabel("Tilted (0/1)")
plt.title("Session Length vs Tilt")
plt.show()

# 2. Losses vs tilt
plt.figure()
plt.scatter(df["session_losses"], df["tilted"])
plt.xlabel("Session Losses")
plt.ylabel("Tilted (0/1)")
plt.title("Losses vs Tilt")
plt.show()

# 3. Role vs tilt
plt.figure()
sns.barplot(x="role", y="tilted", data=df)
plt.title("Tilt Probability by Role")
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
