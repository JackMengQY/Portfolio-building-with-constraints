import pandas as pd
import numpy as np
import cvxpy as cp

# Load the dataset
file_path = "final_corrected_top_performing_stocks_returns.xlsx"
data = pd.read_excel(file_path)

# Ensure 'Date' is in datetime format and set as index
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# Resample to monthly returns
data = data.resample('M').apply(lambda x: (1 + x).prod() - 1)

# Remove duplicate columns (e.g., 'AMR_x', 'AMR_y'), keeping only 'AMR'
data = data.loc[:, ~data.columns.duplicated()]
if "AMR_x" in data.columns or "AMR_y" in data.columns:
    data = data.loc[:, ~data.columns.str.contains("_x|_y")]

# Ensure 'AMR' is present
if "AMR" not in data.columns:
    raise ValueError("The main 'AMR' column was removed unexpectedly.")

# Drop rows with missing values (incomplete months)
data.dropna(inplace=True)

# Calculate mean returns and covariance matrix
mean_returns = data.mean()
cov_matrix = data.cov()

# Define optimization variables
num_securities = len(mean_returns)
weights = cp.Variable(num_securities)

# Define the objective: Minimize portfolio risk (variance)
portfolio_risk = cp.quad_form(weights, cov_matrix)

# Define constraints
expected_return = mean_returns.values @ weights
constraints = [
    cp.sum(weights) == 1,  # Weights must sum to 1
    expected_return >= 0.01,  # At least 1% monthly return
    weights >= 0  # No short selling
]

# Solve the optimization problem
problem = cp.Problem(cp.Minimize(portfolio_risk), constraints)
problem.solve()

# Check solver status and extract results if successful
if problem.status != "optimal":
    print(f"Solver Status: {problem.status}")
    print("Optimization failed. Consider further relaxing constraints or checking data.")
else:
    # Extract results
    optimal_weights = weights.value
    selected_securities = mean_returns.index

    # Create results DataFrame
    results = pd.DataFrame({
        "Security": selected_securities,
        "Optimal Weight": optimal_weights
    })

    # Calculate expected portfolio return and risk
    expected_portfolio_return = np.dot(mean_returns, optimal_weights)
    expected_portfolio_risk = np.sqrt(problem.value)

    # Output results
    print("\nOptimal Portfolio Allocation:")
    print(results)
    print(f"\nExpected Portfolio Return: {expected_portfolio_return:.2%}")
    print(f"Expected Portfolio Risk: {expected_portfolio_risk:.2%}")
