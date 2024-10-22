import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/iris.csv")

species = {'setosa':'red', 'versicolor':'blue', 'virginica':'green'}
#plt.scatter(df['petal_width'], df['petal_length'], c=df['species'].map(species))


df['squared_petal_width'] = df['petal_width']**2
df['squared_petal_length'] = df['petal_length']**2

squared_petal_width = np.asarray(df['squared_petal_width'])
squared_petal_length = np.asarray(df['squared_petal_length'])

print(df)

#plt.show()


n = len(df)

petal_width = np.asarray(df['petal_width'])
petal_length = np.asarray(df['petal_length'])


#find sample width mean
xbar = np.sum(petal_width) / n

# sample length mean 
ybar = np.sum(petal_length) / n

print(xbar, ybar)
print(n)
print(n * (np.sum(squared_petal_width) - np.square(np.sum(petal_width))))
print(n * (np.sum(squared_petal_length) - np.square(np.sum(petal_length))))
print(n * np.sum(petal_length * petal_width) - (np.sum(petal_width) * np.sum(petal_length)))

# sample correlation coefficient
corr_coeff = (np.sum(petal_width * petal_length) - (n * xbar * ybar)) / ((np.sqrt(np.sum(squared_petal_width)) - n * np.square(xbar)) * (np.sqrt(np.sum(squared_petal_length)) - n * np.square(ybar)))
print(corr_coeff)






