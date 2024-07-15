import pandas as pd
import statsmodels.api as sm

# Load the data from CSV file
file_path =(r"C:\Users\kisho\Desktop\big si bucks\MULTTILINEAR  Regression\multiple regression.csv")
print(f"Loading data from: {file_path}")
try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print(df.head())  # Print first few rows to verify data
except Exception as e:
    print(f"Error reading CSV file: {e}")

# Check column names in the DataFrame
print("Columns in DataFrame:")
print(df.columns)

# Ensure column names are correct and exactly match
expected_columns = ['X1', 'X2', 'X3', 'Y']
if not all(col in df.columns for col in expected_columns):
    missing_cols = [col for col in expected_columns if col not in df.columns]
    print(f"Error: Missing columns: {missing_cols}")
else:
    # Separate predictors (X) and response variable (Y)
    X = df[['X1', 'X2', 'X3']]  # Independent variables
    Y = df['Y']                 # Dependent variable

    # Add a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(Y, X).fit()

    # Print the summary of the regression model
    print(model.summary())
