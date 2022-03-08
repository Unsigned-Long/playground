# coding:utf-8

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Ubuntu Mono"
plt.rcParams["font.size"] = 14

x_axis = []
y_axis = []
z_axis = []
i_axis = []
with open("./src.txt", 'r') as file:
    lines = file.readlines()

for line in lines:
    line = line.strip()
    vec = line.split(',')
    x_axis.append(float(vec[0]))
    y_axis.append(float(vec[1]))
    z_axis.append(float(vec[2]))
    i_axis.append(float(vec[3]))

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
plt.scatter(x_axis, y_axis, c=i_axis, s=200, marker='^')

with open("./dst.txt", 'r') as file:
    lines = file.readlines()

x_axis.clear()
y_axis.clear()
z_axis.clear()
i_axis.clear()

for line in lines:
    line = line.strip()
    vec = line.split(',')
    x_axis.append(float(vec[0]))
    y_axis.append(float(vec[1]))
    z_axis.append(float(vec[2]))
    i_axis.append(float(vec[3]))

plt.scatter(x_axis, y_axis, c=i_axis, alpha=0.5)
plt.title("Power[2.0] nearestK[5]")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
# plt.savefig("./img9.png")
plt.show()
