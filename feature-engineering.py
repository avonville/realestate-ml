from __future__ import print_function  # Compatability with Python 3
print( 'Print function ready to serve.' )

# Importing the libraries

# NumPy for numerical computing
import numpy as np
# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.options.mode.chained_assignment = None  # default='warn'
# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# Seaborn for easier visualization
import seaborn as sns

# Load cleaned dataset from Module 2
df = pd.read_csv('cleaned_df.csv')

df.head()

# Creating indicator variable for properties with 2 beds and 2 baths
df['two_and_two'] = ((df.beds == 2) & (df.baths == 2)).astype(int)

# Display percent of rows where two_and_two == 1
df.two_and_two.mean()

# Creating indicator feature for transactions between 2010 and 2013, inclusive
df['during_recession']=df.tx_year.between(2010, 2013).astype(int)

# Percent of transactions where during_recession == 1
df.during_recession.mean()

# Setting variable a as the earlier indicator variable (combining two masks)
a = ((df.tx_year >= 2010)&(df.tx_year <= 2013)).astype(int)

# Set variable b as the new indicator variable (using "between")
b = df.tx_year.between(2010, 2013).astype(int)

# Check for a and b being equivalent
print(all(a==b))

# Creating property age feature
df['property_age'] = df.tx_year - df.year_built

# Should not be less than 0
print(df.property_age.min())

# Number of observations with 'property_age' < 0
print(sum(df.property_age < 0))

# Removing rows where property_age is less than 0
df = df[df.property_age >= 0]

# Number of rows in remaining dataframe
print(len(df))

# Create a school score feature that num_schools * median_school
df['school_score'] = df.num_schools * df.median_school

# Display median school score
df.school_score.median()

# Bar plot for exterior_walls
sns.countplot(y='exterior_walls', data=df)

# Group 'Wood Siding' and 'Wood Shingle' with 'Wood'
df.exterior_walls.replace(['Wood Siding', 'Wood Shingle'], 'Wood', inplace=True)

# List of classes to group
other_exterior_walls = ['Stucco', 'Other', 'Asbestos shingle', 'Concrete Block', 'Masonry', 'Other', 'Concrete', 'Block', 'Rock, Stone']

# Group other classes into 'Other'
df.exterior_walls.replace(other_exterior_walls, 'Other', inplace=True)

# Bar plot for exterior_walls
sns.countplot(y='exterior_walls', data=df)

# Bar plot for roof
sns.countplot(y='roof', data=df)

# Grouping  'Composition' and 'Wood Shake/ Shingles' into 'Composition Shingle'
df.roof.replace(['Composition', 'Wood Shake/ Shingles'], 'Composition Shingle', inplace=True)

# List of classes to group for other
other_roof=['Other', 'Gravel/Rock', 'Roll Composition', 'Slate', 'Built-up', 'Asbestos', 'Metal']

# Grouping other classes into 'Other'
df.roof.replace(other_roof, 'Other', inplace=True)

# Roof Cleaning
df.roof.replace(['shake-shingle', 'asphalt,shake-shingle'], 'Shake Shingle', inplace=True)
df.roof.replace('asphalt', 'Asphalt', inplace=True)
df.roof.replace('composition', 'Composition Shingle', inplace=True)

# Bar plot for roof
sns.countplot(y='roof', data=df)

# Creating new dataframe with dummy features
df = pd.get_dummies(df, columns=['exterior_walls', 'roof', 'property_type'])

# First 5 rows of dataframe with dummies
df.head()

# Drop 'tx_year' and 'year_built' from the dataset
df = df.drop(['tx_year', 'year_built'], axis=1)

# Save analytical base table
df.to_csv('analytical_base_table.csv', index=None)
