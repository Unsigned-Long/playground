# coding:utf-8

import matplotlib.pyplot as plt
import numpy as np

x_data = []
y_data = []
z_data = []

with open("./file4.txt", 'r') as file:
    lines = file.readlines()

x_record = 102
y_record = 50

for line in lines:
    line = line.strip()
    ls = line.split(',')
    x_data.append(float(ls[0]))
    y_data.append(float(ls[1]))
    z_data.append(float(ls[2]))

plt.rcParams["font.family"] = "Ubuntu Mono"
plt.rcParams["font.size"] = 15

plt.scatter(x_data, y_data, c=z_data)

plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("N(6.0, 2.5, 2.5, 3.0, 0.6)")


plt.show()
