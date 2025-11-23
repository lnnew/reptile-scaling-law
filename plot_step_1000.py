import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = {
    'N_tasks': [50, 100, 150, 200, 300, 500, 700, 1000],
    'Accuracy': [0.6504, 0.6403, 0.6496, 0.6320, 0.5997, 0.6139, 0.6256, 0.5861],
    'Loss': [0.9042, 0.9410, 0.9465, 0.9715, 1.0375, 1.0111, 0.9684, 1.0675]
}

df = pd.DataFrame(data)
df = df.sort_values('N_tasks')

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
sns.lineplot(data=df, x='N_tasks', y='Accuracy', marker='o')
plt.title('Step 1000: Meta-Test Accuracy vs N_tasks')
plt.xlabel('Number of Tasks')
plt.ylabel('Accuracy')
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
sns.lineplot(data=df, x='N_tasks', y='Loss', marker='o', color='orange')
plt.title('Step 1000: Meta-Test Loss vs N_tasks')
plt.xlabel('Number of Tasks')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('step_1000_analysis.png')
print("Plot saved to step_1000_analysis.png")
