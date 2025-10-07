# ----------------------
# Module 1 – Setup & EDA
# ----------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
import google.generativeai as genai

from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GENAI_API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")



df = pd.read_csv('/Users/akilkanwar/Desktop/Remaining Usefu Life Prediction for Jet Engines/train_FD001.txt', sep=" ", header=None, engine ='python')

df = df.drop(columns=[26, 27])

columns = ['engine', 'cycle'] + [f'op_setting_{i}' for i in range(3)] + [f'sensor_{i}' for i in range(1, 22)]
df.columns = columns

print(df.head())
print(df.shape)

print(df.isnull().sum())

print(df.describe())

random_engine = random.choice(df['engine'].unique())
engine_data = df[df['engine'] == random_engine]

plt.figure(figsize=(12, 6))
for sensor in ['sensor_1', 'sensor_2', 'sensor_3']:
    plt.plot(engine_data['cycle'], engine_data[sensor], label=sensor)

plt.title(f'Sensor Readings for Engine {random_engine}')
plt.xlabel('Cycle')
plt.ylabel('Sensor Readings')
plt.legend()
plt.show()


sensor_data = df[[f'sensor_{i+1}' for i in range(21)]]
corr_matrix = sensor_data.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Sensor Readings')
plt.show()

# Check for missing values
print(df.isnull().sum())

# Basic statistics
print(df.describe())

# ----------------------
# Module 2 – Add RUL Labels
# ----------------------
max_cpe = df.groupby('engine')['cycle'].max().reset_index()
max_cpe.columns = ['engine', 'max_cycle']

df = df.merge(max_cpe, on='engine', how='left')

df['RUL'] = df['max_cycle'] - df['cycle']

df.drop(columns=['max_cycle'], inplace=True)

print(df.head())
print(df.shape)

plt.figure(figsize=(10,6))
plt.hist(df['RUL'], bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of Remaining Useful Life (RUL)')
plt.xlabel('RUL (cycles)')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# ----------------------
# Module 3 – Train a Simple Regression Model
# ----------------------
corr_with_rul = df.corr()['RUL'].abs().sort_values(ascending=False)
print("Top correlations with RUL:")
print(corr_with_rul.head(10))

top_sensors = corr_with_rul.index[1:6] 
print("Selected sensors:", top_sensors)

X = df[top_sensors]
y = df['RUL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f"Linear Regression MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")


plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')  
plt.ylabel('Predicted RUL')
plt.title('Random Forest: Predicted vs Actual RUL')


#FIX API 
# Replace the Gemini API section

prompt = """
Based on realistic maintenance data for jet engines, estimate:
1. Average cost of unexpected failure (in USD)
2. Average cost of early maintenance (in USD)
3. Reasonable maintenance threshold (in % of predicted RUL)
Please give short numeric results only.
"""

response = model.generate_content(prompt)

print("=== Gemini Output ===")
print(response.text)