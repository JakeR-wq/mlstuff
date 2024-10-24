import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/iris.csv")

df = df.loc[df['species'] == "virginica"]

sepal_length = np.asarray(df['sepal_length'])
petal_length = np.asarray(df['petal_length'])

# find sample variance for one variable
def sample_var(x):
    xbar = np.sum(x) / len(x)
    return (np.sum(np.square(x - xbar))) / (len(x) - 1)

# find pearson correlation coefficient between two variables
def corr_coeff(x, y):
    xbar = np.sum(x) / len(x)
    ybar = np.sum(y) / len(x)  
    return (np.sum(x * y) - (len(x) * xbar * ybar)) / ((np.sqrt(np.sum(np.square(x)) - (len(x) * np.square(xbar)))) * (np.sqrt(np.sum(np.square(y)) - (len(x) * np.square(ybar)))))

# find m and b for the regression line, returns m and b as a list
def regr_line(x, y):
    m = corr_coeff(x,y) * (sample_var(y) / sample_var(x))
    b = (np.sum(y) / len(y)) - (m * (np.sum(x) / len(x)))
    return [m, b]

# plot the data along with the regression line
def graph(x, y):
    regression_line = regr_line(x,y)
    xs = np.linspace(np.min(x), np.max(x), 100)
    ys = (regression_line[0] * xs) + regression_line[1] # y = mx + b
    plt.plot(xs, ys)
    plt.scatter(x, y)
    plt.show()
    return

graph(sepal_length, petal_length)






