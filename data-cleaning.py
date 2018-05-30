
from __future__ import print_function  # Compatability with Python 3
print( 'Print function ready to serve.' )

# Importing of Modules
import pandas as pd
pd.set_option('display.max_columns', 100)
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#Importing Dataset
df = pd.read_csv('project_files/real_estate_data.csv')

# Drop duplicates
df = df.drop_duplicates()
print( df.shape )

# Displaying unique values of 'basement'
df.basement.unique()

# Setting missing basement values to 0
df['basement'] = df.basement.fillna(0)

# Displaying unique values of 'basement'
df.basement.unique()

# Class distributions for 'roof'
sns.countplot(y='roof', data=df)

# Setting 'composition' to 'Composition'
df.roof.replace('composition', 'Composition', inplace=True)

# Setting 'asphalt' to 'Asphalt'
df.roof.replace('asphalt', 'Asphalt', inplace=True)

#  Setting 'shake-shingle' and 'asphalt,shake-shingle' to 'Shake Shingle'
df.roof.replace('shake-shingle', 'Shake-shingle', inplace=True)
df.roof.replace('asphalt,shake-shingle', 'Shake-shingle', inplace=True)

# Class distributions for 'exterior_walls'
sns.countplot(y='exterior_walls', data=df)

# Setting 'Rock, Stone' to 'Masonry'
df.exterior_walls.replace('Rock, Stone', 'Masonry', inplace=True)

# Setting 'Concrete' and 'Block' to 'Concrete Block'
df.exterior_walls.replace(['Concrete', 'Block'], 'Concrete Block', inplace=True)

# Class distributions for 'exterior_walls'
sns.countplot(y='exterior_walls', data=df)

# Box plot of 'tx_price' using the Seaborn library
sns.boxplot(df.tx_price)

# Violin plot of 'tx_price' using the Seaborn library
sns.boxplot(df.tx_price)
plt.xlim(0, 1000000) # setting x-axis range to be the same as in violin plot
plt.show()

sns.violinplot(df.tx_price)
plt.show()

# Violin plot of beds
sns.violinplot(df.beds)
plt.show()
# Violin plot of sqft
sns.violinplot(df.sqft)
plt.show()
# Violin plot of lot_size
sns.violinplot(df.lot_size)
plt.show()

# Sort df.lot_size and display the top 5 samples
df.lot_size.sort_values(ascending=False).head()

# Removing lot_size outliers
df = df[df.lot_size <= 500000]

# print length of df
print(len(df))

# Displaying number of missing values by feature (categorical)
df.select_dtypes(include=['object']).isnull().sum()

# Fill missing categorical values
for column in df.select_dtypes(include=['object']):
    df[column] = df[column].fillna('Missing')

# Display number of missing values by feature (categorical)
df.select_dtypes(include=['object']).isnull().sum()

#Checking Datatypes
df.dtypes

# Displaying number of missing values by feature (numeric)
df.select_dtypes(include=['int64']).isnull().sum()

# Save cleaned dataframe to new file
df.to_csv('cleaned_df.csv', index=None)
