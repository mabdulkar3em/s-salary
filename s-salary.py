import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('Salary_dataset.csv')
data = data.drop(["Unnamed: 0"], axis=1)  # Changed from "Unnamed :0" to "Unnamed: 0" (removed space)
x =data['YearsExperience'].values.reshape(-1, 1)
y= data['Salary'].values.reshape(-1, 1)
# Display the first few rows of the dataset
print(data.info())
print(33 * "$$")

model = LinearRegression()
model.fit(x, y)

f= model.score(x,y)
z= model.predict([[2.1]])
print("Predicted Salary for 2.1 years of experience:", z)

# Visualize the data
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(x='YearsExperience', y='Salary', data=data, color='blue', alpha=0.7)
plt.title('Salary vs Years of Experience', fontsize=16)
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Salary ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Add regression line

if 'model' in locals():
    X_range = np.linspace(data['YearsExperience'].min(), data['YearsExperience'].max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_range)
    plt.plot(X_range, y_pred, color='red', linewidth=2, label=f'Regression Line (R² = {f:.2f})')
    plt.legend()

# Add annotations for some data points
for i in range(min(5, len(data))):
    plt.annotate(f'({data["YearsExperience"].iloc[i]}, ${data["Salary"].iloc[i]:,})',
                xy=(data["YearsExperience"].iloc[i], data["Salary"].iloc[i]),
                xytext=(12, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.tight_layout()
plt.savefig('salary_experience_plot.png', dpi=300)
plt.show()
