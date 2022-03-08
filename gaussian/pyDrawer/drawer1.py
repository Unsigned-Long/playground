# coding:utf-8

import os
import matplotlib.pyplot as plt

# read data
file = open("./pyDrawer/file1.txt", 'r')
x1, y1 = [], []
data = file.readlines()

for strLine in data:
    dataLine = strLine.strip().split(',')
    if len(dataLine) == 0:
        continue
    x1.append(float(dataLine[0]))
    y1.append(float(dataLine[1]))
file.close()
# read data
file = open("./pyDrawer/file2.txt", 'r')
x2, y2 = [], []
data = file.readlines()

for strLine in data:
    dataLine = strLine.strip().split(',')
    if len(dataLine) == 0:
        continue
    x2.append(float(dataLine[0]))
    y2.append(float(dataLine[1]))
file.close()


fig, ax = plt.subplots()
ax.scatter(x1, y1, marker='o', label="N(7.0,2.0)", c='#d62728')
ax.scatter(x2, y2, marker='^', label="N(5.0,5.0)", c='#2ca02c')
ax.legend(loc="upper left")
ax.plot(x1, y1, c='#d62728')
ax.plot(x2, y2, c='#2ca02c')

plt.title("Gaussian[One Diem]")
plt.show()
