from __future__ import print_function  # Compatability with Python 3
print( 'Print function ready to serve.' )

# Importing the libraries
# NumPy for numerical computing
import numpy as np
# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# Seaborn for easier visualization
import seaborn as sns
# Scikit-Learn for Modeling
import sklearn
# Import Elastic Net, Ridge Regression, and Lasso Regression
from sklearn.linear_model import ElasticNet, Ridge, Lasso
# Import Random Forest and Gradient Boosted Trees
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# Function for splitting training and test set
from sklearn.model_selection import train_test_split # Scikit-Learn 0.18+
# Function for creating model pipelines
from sklearn.pipeline import make_pipeline
# For standardization
from sklearn.preprocessing import StandardScaler
# Helper for cross-validation
from sklearn.model_selection import GridSearchCV
# Function for Checking fitted modules
from sklearn.exceptions import NotFittedError
# Import r2_score and mean_absolute_error functions
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Loading cleaned dataset
df = pd.read_csv('analytical_base_table.csv')
print(df.shape)
df.mean()

# Creating separate object for target variable
y = df.tx_price
# Creating separate object for input features
X = df.drop('tx_price', axis=1)

# Spliting X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print( len(X_train), len(X_test), len(y_train), len(y_test) )

# Summary statistics of X_train
X_train.describe()

# Standardize X_train
X_train_new = (X_train - X_train.mean()) / X_train.std()

# Summary statistics of X_train_new
X_train_new.describe()

# Create pipelines dictionary
pipelines = {
    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=123)),
    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=123)),
    'enet'  : make_pipeline(StandardScaler(), ElasticNet(random_state=123))
}

# Add a pipeline for 'rf'
pipelines['rf'] = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123))

# Add a pipeline for 'gb'
pipelines['gb'] = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123) )

# Checking all 5 algorithms = pipelines
for key, value in pipelines.items():
    print( key, type(value) )

# Lasso hyperparameters
lasso_hyperparameters = {
    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
}

# Ridge hyperparameters
ridge_hyperparameters = {
    'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
}

# Elastic Net hyperparameters
enet_hyperparameters = {
    'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Random forest hyperparameters
rf_hyperparameters = {
    'randomforestregressor__n_estimators': [100, 200],
    'randomforestregressor__max_features': ['auto', 'sqrt', 0.33],
}

# Boosted tree hyperparameters
gb_hyperparameters = {
    'gradientboostingregressor__n_estimators': [100, 200],
    'gradientboostingregressor__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [1, 3, 5],
}

# Create hyperparameters dictionary
hyperparameters = {
    'rf' : rf_hyperparameters,
    'lasso' : lasso_hyperparameters,
    'enet' : enet_hyperparameters,
    'ridge' : ridge_hyperparameters,
    'gb' : gb_hyperparameters,
}

# Checking list of hyperparameters
for key in ['enet', 'gb', 'ridge', 'rf', 'lasso']:
    if key in hyperparameters:
        if type(hyperparameters[key]) is dict:
            print( key, 'was found in hyperparameters, and it is a grid.' )
        else:
            print( key, 'was found in hyperparameters, but it is not a grid.' )
    else:
        print( key, 'was not found in hyperparameters')

#Checking to make sure their are no Null values
print( np.any(np.isnan(X)) )

print( np.all(np.isfinite(X)) )

# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    model.fit(X_train, y_train)
    fitted_models[name] = model
    print(name, 'has been fitted.')

# Checking 5 cross-validation objects
for key, value in fitted_models.items():
    print( key, type(value) )

#Checking for models being correctly fitted
from sklearn.exceptions import NotFittedError

for name, model in fitted_models.items():
    try:
        pred = model.predict(X_test)
        print(name, 'has been fitted.')
    except NotFittedError as e:
        print(repr(e))

# Display best_score_ for each fitted model
for name, model in fitted_models.items():
    print( name, model.best_score_ )

# Display fitted random forest object
fitted_models['rf']

# Predict test set using fitted random forest
pred = fitted_models['rf'].predict(X_test)

# Calculating R^2 and MAE
print( 'R^2:', r2_score(y_test, pred ))
print( 'MAE:', mean_absolute_error(y_test, pred))

# Evaluating fitted modules
for name, model in fitted_models.items():
    pred = model.predict(X_test)
    print( name )
    print( '--------' )
    print( 'R^2:', r2_score(y_test, pred ))
    print( 'MAE:', mean_absolute_error(y_test, pred))
    print()

# Plotting winning module
gb_pred = fitted_models['rf'].predict(X_test)
plt.scatter(gb_pred, y_test)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()


# Checking datatypes
type(fitted_models['rf'])
type(fitted_models['rf'].best_estimator_)


fitted_models['rf'].best_estimator_

import pickle

with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)
