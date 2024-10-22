import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/iris.csv")

species = {'setosa':'red', 'versicolor':'blue', 'virginica':'green'}
#plt.scatter(df['petal_width'], df['petal_length'], c=df['species'].map(species))


df['squared_petal_width'] = df['petal_width']**2
df['squared_petal_length'] = df['petal_length']**2

print(df)

#plt.show()


n = len(df)

#find sample width mean
xbar = df['petal_width'].sum() / n

# sample length mean 
ybar = df['petal_length'].sum() / n


print(xbar, ybar)

# sample correlation coefficient
corr_coeff = (n * (df['petal_length'].sum() * df['petal_width'].sum()) - (df['petal_width'].sum() * df['petal_length'].sum()) / (
                np.sqrt(n * df['squared_petal_width'].sum() - np.square(df['petal_width'].sum())) * 
                np.sqrt(n * df['squared_petal_length'].sum() - np.square(df['petal_length'].sum()))))
print(corr_coeff)




