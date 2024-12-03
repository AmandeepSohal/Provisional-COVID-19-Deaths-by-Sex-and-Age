import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load your data
analysis_data = pd.read_csv("C:/Users/Amandeep Sohal/Desktop/College/University Years/Masters/Stat Methods Final/Provisional_COVID-19_Deaths_by_Sex_and_Age_20241116.csv")

# Convert 'Age Group' to a numerical representation (excluding 'All Ages')
age_group_map = {
    'Under 1 year': 0.5,
    '1-4 years': 2.5,
    '5-14 years': 9.5,
    '15-24 years': 19.5,
    '25-34 years': 29.5,
    '35-44 years': 39.5,
    '45-54 years': 49.5,
    '55-64 years': 59.5,
    '65-74 years': 69.5,
    '75-84 years': 79.5,
    '85 years and over': 92.5
}

# Exclude rows where 'Age Group' is 'All Ages'
analysis_data = analysis_data[analysis_data['Age Group'] != 'All Ages']

# Map 'Age Group' to numeric values
analysis_data['Age Group Numeric'] = analysis_data['Age Group'].map(age_group_map)

# Drop rows where 'Age Group Numeric' is NaN (if there are unmapped categories)
analysis_data.dropna(subset=['Age Group Numeric'], inplace=True)

# Drop rows where 'COVID-19 Deaths' is NaN
analysis_data.dropna(subset=['COVID-19 Deaths'], inplace=True)

# Remove outliers in 'COVID-19 Deaths' using the IQR method
q1 = analysis_data['COVID-19 Deaths'].quantile(0.25)
q3 = analysis_data['COVID-19 Deaths'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
print(q1, q3, iqr, lower_bound, upper_bound)
analysis_data = analysis_data[(analysis_data['COVID-19 Deaths'] >= lower_bound) & 
                               (analysis_data['COVID-19 Deaths'] <= upper_bound)]

# Verify data after cleaning
print(f"Data shape after removing missing values and outliers: {analysis_data.shape}")

# Define features and target
X = analysis_data[['Age Group Numeric']]
y = analysis_data['COVID-19 Deaths']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot actual vs predicted
# Enhanced scatter plot: Actual vs Predicted
# Clean scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 6))

# Plot actual data

sns.scatterplot(x=X_test['Age Group Numeric'], y=y_test, 
                s=60, label='Actual Data', alpha=0.8)

# Plot predicted data
sns.scatterplot(x=X_test['Age Group Numeric'], y=y_pred,
                color = "orange", s=80, label='Predicted Data', alpha=0.8)

# Add regression line
sns.lineplot(x=X_test['Age Group Numeric'], y=y_pred,
             color = "black", linewidth=3, label='Regression Line')

# Customize plot appearance
plt.xlabel('Age Group Numeric', fontsize=14)
plt.ylabel('COVID-19 Deaths', fontsize=14)
plt.title('Linear Regression: Age Group vs COVID-19 Deaths', fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # Subtle gridlines
plt.legend(fontsize=12, loc='best', frameon=False)  # Simplified legend
plt.tight_layout()

# Remove excess borders
sns.despine()

# Display plot
plt.tight_layout()
plt.show()
