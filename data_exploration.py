import pandas as pd

# Load dataset
df = pd.read_csv("data/ckd.csv")

# Display first few rows
print(df.head())

# Check dataset info
print(df.info())

# Check for missing values
print(df.isnull().sum())