#%% md
# # Titanic DataAnalysis
#%%
import pandas as pd

# Function to load data
def load_data(file_path):

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


file_path = "/Users/pratigyajamakatel/Downloads/Titanic-Dataset.csv"
df = load_data(file_path)

# Check if data is loaded
if df is not None:
    print(df.head())

#%%

#%%

#%% md
# # Display Basic info
#%%
def basic_info(df):
    print("\nDataset Info:")
    print(df.info())


# If data loaded successfully, display basic info
if df is not None:
    basic_info(df)

#%% md
# # Checking Missing Values
#%%
def missing_values(df):
    print("\n🔹 Missing Values:")
    print(df.isnull().sum())

# Check missing values in the dataset
if df is not None:
    missing_values(df)

#%% md
# # Fill missing values
#%%
def handle_missing_values(df):
    df["Age"].fillna(df["Age"].median())
    df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Cabin"].fillna("Unknown")
    return df

# Handle missing values
if df is not None:
    df = handle_missing_values(df)
    print("\n Missing values handled!")

#%% md
# # Statistical Summary
#%%
def summary_statistics(df):
    print("\n🔹 Summary Statistics:")
    print(df.describe())

# Display statistical summary
if df is not None:
    summary_statistics(df)

#%% md
# # Visualizing missing data
#%%
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_missing_data(df):
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
    plt.title("Missing Data Heatmap")
    plt.show()

# Visualize missing data
if df is not None:
    visualize_missing_data(df)

#%% md
# # Data Distribution
#%%
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_missing_data(df):
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
    plt.title("Missing Data Heatmap")
    plt.show()

# Visualize missing data
if df is not None:
    visualize_missing_data(df)

#%% md
# # Survival vs NotSurvioval
#%%
# Define the survival_count function correctly
def survival_count(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Survived",hue="Sex",data=df, palette="Set2")
    plt.title("Survival Count")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.show()

# Visualize survival count
if df is not None:
    survival_count(df)


#%% md
# # Survival Rate By Gender
#%%
def survival_by_gender(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Sex", hue="Survived", data=df, palette="Set1")
    plt.title("Survival by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.legend(["Not Survived", "Survived"])
    plt.show()

# Visualize survival rate by gender
if df is not None:
    survival_by_gender(df)

#%% md
# # Survival Rate by Passenger Class
#%%
def survival_by_class(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Pclass", hue="Survived", data=df, palette="coolwarm")
    plt.title("Survival by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Count")
    plt.legend(["Not Survived", "Survived"])
    plt.show()

# Visualize survival rate by passenger class
if df is not None:
    survival_by_class(df)

#%% md
# # Correlation Heatmap
#%%
# Define the correlation_heatmap function
def correlation_heatmap(df):
    # Filter numeric columns only
    df_numeric = df.select_dtypes(include=[np.number])  # Keep only numeric columns

    # Plotting the correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()


# Display correlation heatmap if df is not None
if df is not None:
    correlation_heatmap(df)

#%% md
# # Feature Engineering
#%%
def feature_engineering(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print("\n🔹 Feature 'FamilySize' added!")
    return df

# Add family size feature
if df is not None:
    df = feature_engineering(df)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Feature Engineering: Creating new column "FamilySize
def feature_engineering(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print("\n🔹 Feature 'FamilySize' added!")
    return df

# Visualize FamilySize distribution
def visualize_family_size(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(df["FamilySize"], bins=20, kde=True, color='purple')
    plt.title("Family Size Distribution")
    plt.xlabel("Family Size")
    plt.ylabel("Count")
    plt.show()

# Run feature engineering and visualization
if df is not None:
    df = feature_engineering(df)  # Perform feature engineering
    visualize_family_size(df)    # Visualize Family Size Distribution

#%% md
# # Age Distribution
#%%
def visualize_age_distribution(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(df["Age"], bins=30, kde=True, color='green')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

# Run the age distribution visualization
if df is not None:
    visualize_age_distribution(df)

#%% md
# # Survival vs Fare
#%%
def survival_by_fare(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Survived",hue="Sex", y="Fare", data=df, palette="Set2")
    plt.title("Survival vs. Fare")
    plt.xlabel("Survival (0 = No, 1 = Yes)")
    plt.ylabel("Fare")
    plt.show()

# Run the survival vs fare visualization
if df is not None:
    survival_by_fare(df)

#%% md
# 
#%%
def drop_unnecessary_columns(df):
    # Example: Drop columns with too many missing values or irrelevant columns
    columns_to_drop = ['Fare']  # Adjust based on visualizations
    df = df.drop(columns=columns_to_drop, axis=1)
    print("\n Unnecessary columns dropped!")
    return df

# Drop unnecessary columns
if df is not None:
    df = drop_unnecessary_columns(df)
