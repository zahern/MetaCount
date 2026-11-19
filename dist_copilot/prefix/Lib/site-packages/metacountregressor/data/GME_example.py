#how to check what version of python
import sys

print("Python version:", sys.version)
import gme
import pandas as pd
import numpy as np
import  statsmodels.api as sm
import matplotlib.pyplot as plt
gravity_data = pd.read_csv('https://www.usitc.gov/data/gravity/example_trade_and_grav_data_small.csv')

gravity_gc = pd.read_csv('Alterynx merged dataset_v3.csv')
gravity_gc['year'] = 2018
gravity_gc['sqrt_distance'] = np.sqrt(gravity_gc['Avg_Road_OP_Dist']+1)
gravity_gc['ln_distance'] = np.log(gravity_gc['Avg_Road_OP_Dist']+1)
gravity_gc['ln_pop'] = np.log(gravity_gc['POPHH']+1)
gravity_gc['Inc'] = np.log(gravity_gc['Avg Medium Income']+1

)
gravity_gc['one'] =1

gravity_gc.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check for NaN values in all columns
print(gravity_gc.isnull().sum())

# Verify if there are any NaN values in the dependent and independent variables
print(gravity_gc[['Ticket', 'Avg_Road_OP_Dist', 'Avg Medium Income', 'Avg_PT_OP_Time', 'Avg_Road_OP_Time', 'Accommodation_Rooms']].isnull().sum())

#gravity_gc['Ticket '] = gravity_gc['Ticket'].astype(int)
#clean gravity_gc to make sure its not nans
#poisson_model = sm.GLM(gravity_gc['Ticket'], gravity_gc[['one', 'sqrt_distance', 'Avg_PT_OP_Time', 'Avg_Road_OP_Time']], offset=gravity_gc['POPHH'], family=sm.families.Poisson()).fit()
#print(poisson_model.summary())
#nb2_training_results = sm.GLM(gravity_gc['Ticket'], gravity_gc[['one', 'Avg_Road_OP_Dist', 'Avg Medium Income', 'Avg_PT_OP_Time', 'Avg_Road_OP_Time', 'Accommodation_Rooms']], offset=1/gravity_gc['ln_pop'], family = sm.families.NegativeBinomial()).fit()
#print(nb2_training_results.summary())

# Next, we use the loaded data to create an EstimationData instance called gme_data
gme_data = gme.EstimationData(data_frame=gravity_data,
                              imp_var_name='importer',
                              exp_var_name='exporter',
                              trade_var_name='trade_value',
                              year_var_name='year',
                              notes='Downloaded from https://www.usitc.gov/data/gravity/example_trade_and_grav_data_small.csv')

gme_data_gc = gme.EstimationData(data_frame=gravity_gc, imp_var_name='SA2_NAME21', exp_var_name='Sport', trade_var_name = 'Ticket', year_var_name=
                                 'year')

model = gme.EstimationModel(estimation_data= gme_data_gc, lhs_var = 'Ticket', rhs_var= ['one', 'sqrt_distance', 'Avg Medium Income', 'ln_pop',
                                                                                                   ], fixed_effects = [
    ['Sport']
])
#model = gme.EstimationModel(estimation_data= gme_data_gc, lhs_var = 'Ticket', rhs_var= ['Avg_Road_OP_Dist', 'Avg Medium Income', 'Avg_PT_OP_Time', 'Accommodation_Rooms'])

estimates = model.estimate()
results = estimates['all']
results.summary()
print(results.summary())


# Rescale the predictors
gravity_gc['sqrt_distance'] = (gravity_gc['sqrt_distance']/gravity_gc['sqrt_distance'].mean())
gravity_gc['Avg Medium Income'] = (gravity_gc['Avg Medium Income']/gravity_gc['Avg Medium Income'].mean())
gravity_gc['ln_pop'] = (gravity_gc['ln_pop']/gravity_gc['ln_pop'].mean())



lhs_var = 'Ticket'
rhs_vars = ['sqrt_distance', 'Avg Medium Income', 'ln_pop']
fixed_effects_var = 'Sport'

# Ensure dependent variable is numeric
y = pd.to_numeric(gravity_gc[lhs_var], errors='coerce')

# Create independent variables (numeric predictors)
X = gravity_gc[rhs_vars]
XX = X.copy()
# Convert the 'Sport' column (strings) into dummy variables
sports_dummies = pd.get_dummies(gravity_gc[fixed_effects_var], prefix=fixed_effects_var)
# Drop the column 'Sport_Swimming' from the sports_dummies DataFrame
sports_dummies = sports_dummies.drop(columns=['Sport_Swimming'])
# Add dummy variables for sports to the independent variables
X = pd.concat([X, sports_dummies], axis=1)

# Add an intercept term
X['Intercept'] = 1

# Handle missing values
X = X.apply(pd.to_numeric, errors='coerce')  # Ensure all columns are numeric
X = X.fillna(0)                             # Fill missing values
y = y.fillna(0)                             # Fill missing values
print(X.head())
# Ensure dimensions of X and y match

# Fit the linear regression model
for col in X.select_dtypes(include=['bool']):
    X[col] = X[col].astype(int)
print("X Shape:", X.shape)
print("y Shape:", y.shape)
#zero inflated

results = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha = 1)).fit()
results_s = sm.GLM(y, XX, family=sm.families.NegativeBinomial(alpha =1)).fit()
print(results.summary())
model = sm.OLS(y, XX).fit()

# Print the summary
print(model.summary())




# Step 1: Extract coefficients from both models
# For gme.EstimationModel
gme_coefficients = results.params  # Extract coefficients from gme model
gme_coefficients.name = 'Venue Poisson Model'
sm_coefficients = results_s.params
sm_coefficients.name = 'Poisson Model'

# For sm.OLS
ols_coefficients = model.params  # Extract coefficients from OLS model
ols_coefficients.name = 'OLS Model'

# Combine coefficients into a single DataFrame for comparison
coefficients_comparison = pd.concat([gme_coefficients, sm_coefficients, ols_coefficients], axis=1)

# Step 2: Standardize coefficients for comparability
coefficients_comparison_standardized = coefficients_comparison / coefficients_comparison.abs().max()

# Step 3: Plot the coefficients
coefficients_comparison_standardized.plot(kind='bar', figsize=(10, 6))
plt.title('Comparison of Standardized Coefficients')
plt.xlabel('Variables')
plt.ylabel('Standardized Coefficient Value')
plt.legend(['Venue Poisson Model', 'Poisson Model', 'OLS Model'])
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 4: Compare residuals
# Compute residuals for both models
gme_residuals = y.squeeze() - results.predict(X)
sm_residuals = y.squeeze() - results_s.predict(XX)
ols_residuals = y.squeeze() - model.predict(XX)

# Plot residuals
plt.figure(figsize=(10, 6))
plt.hist(gme_residuals, bins=30, alpha=0.5, label=' Venue Model Residuals')
plt.hist(sm_residuals, bins=30, alpha=0.5, label='NB Model Residuals')
plt.hist(ols_residuals, bins=30, alpha=0.5, label=' Model Residuals')
plt.title('Comparison of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('residuals_comparison.png', dpi=300)  # Save as PNG with 300 DPI
plt.show()

# Step 5: Compare predicted vs. actual values
# Predicted values
gme_predictions = results.predict(X)
sm_predictions = results_s.predict(XX)
ols_predictions = model.predict(XX)

# Plot predicted vs. actual
plt.figure(figsize=(10, 6))
plt.scatter(y.squeeze(), gme_predictions, alpha=0.5, label='Venue Model')
plt.scatter(y.squeeze(), sm_predictions, alpha=0.5, label='NB Model')
plt.scatter(y.squeeze(), ols_predictions, alpha=0.5, label='OLS Model')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Fit')
plt.title('Predicted vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('predicted_comparison.png', dpi=300)  # Save as PNG with 300 DPI
plt.show()

# Step 6: Compare performance metrics
from sklearn.metrics import mean_squared_error, r2_score

# Compute MSE and R-squared for both models
gme_mse = mean_squared_error(y, gme_predictions)
sm_mse = mean_squared_error(y, sm_predictions)
ols_mse = mean_squared_error(y, ols_predictions)

gme_r2 = r2_score(y, gme_predictions)
sm_r2 = r2_score(y, sm_predictions)
ols_r2 = r2_score(y, ols_predictions)

print("Performance Metrics:")
print(f"Venue Model - MSE: {gme_mse:.4f}, R-squared: {gme_r2:.4f}")
print(f"NB Model - MSE: {sm_mse:.4f}, R-squared: {sm_r2:.4f}")
print(f"OLS Model - MSE: {ols_mse:.4f}, R-squared: {ols_r2:.4f}")
print(1)




# Define the zero-inflated Poisson model
# 'exog_infl' specifies predictors for the zero-inflation model
model_zip = sm.ZeroInflatedPoisson(
    endog=y,  # Dependent variable
    exog=sm.add_constant(XX),  # Predictors for the count process
    exog_infl=sm.add_constant(XX['sqrt_distance']),  # Predictors for the zero-inflation process (can match exog)
    inflation='logit'  # Use logit for the inflation model
)

# Fit the ZIP model
results_zip = model_zip.fit(method='bfgs', maxiter=500, disp=True)

# Print the summary of the ZIP model
print("\nZero-Inflated Poisson Model Summary:")
print(results_zip.summary())

# Generate predictions
predicted_counts = results_zip.predict(sm.add_constant(XX))

# Add predictions to the original data for comparison

# Scatter plot of observed vs. predicted
plt.figure(figsize=(8, 6))
plt.scatter(y.squeeze(),predicted_counts, alpha=0.6)
plt.plot([0, max(y)], [0, max(y)], color='red', linestyle='--')  # 45-degree line
plt.xlabel('Observed Counts')
plt.ylabel('Predicted Counts')
plt.title('Observed vs. Predicted Counts')
plt.grid(True)
plt.savefig('zipng.png')
plt.show()



from pymer4.models import Lmer

# Fit a Poisson GLMM with random intercept for Sport
model = Lmer(
    formula="Ticket ~ sqrt_distance + Avg_Medium_Income + ln_pop + (1|Sport)",
    data=gravity_gc,
    family="poisson"  # Specify Poisson family
)

# Fit the model
result = model.fit()



# Print results
print(result.summary())
gravity_gc['predicted'] = result.predict()
plt.figure(figsize=(8, 6))
plt.scatter(gravity_gc['Ticket'], gravity_gc['predicted'], alpha=0.6, color='blue', label='Data')
plt.plot([0, max(gravity_gc['Ticket'])], [0, max(gravity_gc['Ticket'])], color='red', linestyle='--', label='Perfect Fit')  # 45-degree line
plt.xlabel('Observed Counts')
plt.ylabel('Predicted Counts')
plt.title('Observed vs. Predicted Counts (GLMM)')
plt.legend()
plt.grid(True)
plt.savefig('mixed.png')
plt.show()