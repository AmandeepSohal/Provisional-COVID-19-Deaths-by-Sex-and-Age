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

# Drop rows with missing values in relevant columns
analysis_data = analysis_data[analysis_data['Sex'] != 'All Sexes']
analysis_data = analysis_data.dropna(subset=['COVID-19 Deaths', 'Sex', 'Age Group'])

# 1. T-test: Gender comparison
male_deaths = analysis_data[analysis_data["Sex"] == "Male"]["COVID-19 Deaths"]
female_deaths = analysis_data[analysis_data["Sex"] == "Female"]["COVID-19 Deaths"]

# Perform t-test
t_test_result = stats.ttest_ind(male_deaths, female_deaths, nan_policy='omit')
print("t-test result (gender comparison):", t_test_result)

# Plot boxplot for t-test (Gender comparison)
##sns.boxplot(x="Sex", y="COVID-19 Deaths", data=analysis_data, palette="Set2", linewidth=2)
sns.set_theme()
sns.barplot(data=analysis_data, x="Sex", y="COVID-19 Deaths", hue = "Sex")
plt.title("COVID-19 Deaths by Gender (Colored by " + "sex" + ")", fontsize=15)
plt.xlabel("Gender", fontsize=14)
plt.ylabel("COVID-19 Deaths", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add numbers above each bar
for p in plt.gca().patches:
    height = p.get_height()
    x_coord = p.get_x() + p.get_width() / 2
    y_coord = height / 2  # Position text at the middle of the bar
    plt.text(x_coord, y_coord, f'{height:.0f}', ha='center', va='center')

plt.tight_layout()
plt.show()