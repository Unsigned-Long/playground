# coding:utf-8

import os
import matplotlib.pyplot as plt


# read data
file = open("./pyDrawer/file3.txt", 'r')
x1, y1, z1 = [], [], []
data = file.readlines()

for strLine in data:
    dataLine = strLine.strip().split(',')
    if len(dataLine) == 0:
        continue
    x1.append(float(dataLine[0]))
    y1.append(float(dataLine[1]))
    z1.append(float(dataLine[2]))
file.close()
# read data
file = open("./pyDrawer/file4.txt", 'r')
x2, y2, z2 = [], [], []
data = file.readlines()

for strLine in data:
    dataLine = strLine.strip().split(',')
    if len(dataLine) == 0:
        continue
    x2.append(float(dataLine[0]))
    y2.append(float(dataLine[1]))
    z2.append(float(dataLine[2]))
file.close()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, y1, z1, label="N(5.0,2.5,2.5,1.5,0.3)")
ax.scatter(x2, y2, z2, label="N(6.0,4.5,1.5,1.0,0.6)")
ax.legend(loc="upper left")
plt.title("Gaussian[Two Diem]")


plt.show()
