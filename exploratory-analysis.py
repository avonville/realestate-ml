# Compatability with Python 3
from __future__ import print_function
print( 'Print function ready to serve.' )
# NumPy for numerical computing
import numpy as np
# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)
# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots
get_ipython().run_line_magic('matplotlib', 'inline')
# Seaborn for easier visualization
import seaborn as sns

# Loading real estate data from CSV
df = pd.read_csv('project_files/real_estate_data.csv')

# Dataframe dimensions
df.shape

# Column datatypes
df.dtypes

# Displaying first 5 rows of df
df.head()

# Filter and displaying only df.dtypes that are 'object'
df.dtypes[df.dtypes == 'object']

# Looping through categorical feature names and print each one
for feature in df.dtypes[df.dtypes == 'object'].index:
    print(feature)

# Displaying the first 10 rows of data
df.head(10)

# Displaying last 5 rows of data
df.tail()

#Histogram grid
df.hist(figsize=(14,14), xrot=-45)
# Clearing the text residue
plt.show()

# Summarizing numerical features
df.describe()

# Summarizing categorical features
df.describe(include=['object'])

# Plot bar plot for each categorical feature
for feature in df.dtypes[df.dtypes == 'object'].index:
    sns.countplot(y=feature, data=df)
    plt.show()

# Segmentting tx_price by property_type and plot distributions
sns.boxplot(y='property_type', x='tx_price', data=df)
plt.show()

# Segmentting by property_type and display the means within each class
df.groupby('property_type').mean()

# Segment sqft by sqft and property_type distributions
sns.boxplot(y='property_type', x='sqft', data=df)
plt.show()

# Segment by property_type and display the means and standard deviations within each class
df.groupby('property_type').agg(['mean', 'std'])

# Calculating correlations between numeric features
correlations = df.corr()

# Visualize the correlation grid with a heatmap.
plt.figure(figsize=(7,6))
sns.heatmap(correlations)

#Color scheme
sns.set_style("white")
plt.figure(figsize=(10,8))
sns.heatmap(correlations)

# Plot heatmap of annotated correlations
sns.heatmap(correlations * 100, annot=True, fmt='.0f')

# Generating a mask for the upper triangle
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Plot heatmap of correlations with mask
sns.heatmap(correlations * 100, annot=True, fmt='.0f', mask=mask)

# Plot heatmap of correlations with mask and without cbar
sns.heatmap(correlations * 100, annot=True, fmt='.0f', mask=mask, cbar=False)
