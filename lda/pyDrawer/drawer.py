from cProfile import label
import matplotlib.pyplot as plt

import csv

plt.rcParams["font.family"] = "Ubuntu Mono"
plt.rcParams["font.size"] = 13

filename = "./data/data_after.csv"

x1 = [[], [], [], []]
x2 = [[], [], [], []]

with open(filename) as file:
    lines = csv.reader(file)
    for line in lines:
        if line[0] == "Q1":
            idx = 0
        elif line[0] == "Q2":
            idx = 1
        elif line[0] == "Q3":
            idx = 2
        elif line[0] == "Q4":
            idx = 3
        x1[idx].append(float(line[1]))
        x2[idx].append(float(line[2]))


plt.scatter(x1[0], x2[0], label="Q1")
plt.scatter(x1[1], x2[1], label="Q2")
plt.scatter(x1[2], x2[2], label="Q3")
plt.scatter(x1[3], x2[3], label="Q4")

plt.xlabel('X1')
plt.ylabel('X2')

plt.legend()

plt.show()
