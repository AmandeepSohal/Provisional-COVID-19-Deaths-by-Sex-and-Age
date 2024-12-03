# 2. ANOVA: Age group comparison
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your data
analysis_data = pd.read_csv("C:/Users/Amandeep Sohal/Desktop/College/University Years/Masters/Stat Methods Final/Provisional_COVID-19_Deaths_by_Sex_and_Age_20241116.csv")
analysis_data = analysis_data.dropna(subset=['COVID-19 Deaths', 'Sex', 'Age Group'])

desired_age_groups = [
    'Under 1 year', '1-4 years', '5-14 years', '15-24 years', '25-34 years',
    '35-44 years', '45-54 years', '55-64 years', '65-74 years', '75-84 years',
    '85 years and over']
filtered_data = analysis_data[analysis_data["Age Group"].isin(desired_age_groups)]

# Calculate ANOVA excluding missing values
age_groups = [group["COVID-19 Deaths"].dropna() for name, group in filtered_data.groupby("Age Group")]
anova_result = stats.f_oneway(*age_groups)
print("ANOVA result (age group comparison - filtered):", anova_result)

# Plot barplot (using filtered data)
# Estimate data width (adjust buffer as needed)
num_categories = len(desired_age_groups)
estimated_width = num_categories + 4  # Add buffer for labels and margins

# Set figure width based on data width
plt.figure(figsize=(estimated_width, 6))

sns.barplot(data=filtered_data, x="Age Group", y="COVID-19 Deaths", hue="Age Group")
plt.title("COVID-19 Deaths by Age Group (Filtered)", fontsize=15)
plt.xlabel("Age Group", fontsize=10)
plt.ylabel("COVID-19 Deaths", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=14)

# Add numbers above each bar
for p in plt.gca().patches:
    height = p.get_height()
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = height / 2  # Position text at the middle of the bar
    plt.text(x_coord, y_coord, f'{height:.0f}', ha='center', va='center')

plt.tight_layout()
plt.show()
