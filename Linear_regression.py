import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('student_scores.csv')

def gradient_descent(m_now , b_now,points, alpha):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].Hours
        y = points.iloc[i].Scores
        m_gradient += -(2/n) * x * (y - (m_now*x + b_now))
        b_gradient += -(2/n) * (y - (m_now*x + b_now))    
    m = m_now - m_gradient * alpha
    b = b_now - b_gradient * alpha   
    return m,b
m = 0
b = 0
alpha = 0.01
epochs = 1000  #number of iterations

for i in range(epochs):
    if i % 10 ==0:
        print(f"Epoch:{i}")
    m, b = gradient_descent(m, b,data,alpha)
print(m,b)
plt.scatter(data.Hours,data.Scores, color="black")
x_vals = np.linspace(min(data.Hours), max(data.Hours), 100)
y_vals = [m * x + b for x in x_vals]
plt.plot(x_vals, y_vals, color="red")
plt.show()