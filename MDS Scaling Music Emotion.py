#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set(style='ticks')
np.set_printoptions(precision=3)
pd.set_option("display.precision", 3)
# plt.rcParams["figure.figsize"] = [6, 6]
# plt.rcParams["figure.autolayout"] = True
# overlapping = 0.9
from tempfile import TemporaryFile


# In[2]:


fname = "/Volumes/CHAIZHI/G25 PRD lab/brownbag/01:13:2022/music.csv"
music = pd.read_csv(fname, float_precision='round_trip').to_numpy()
print(music)
print(np.shape(music));print()

path = "/Volumes/CHAIZHI/G25 PRD lab/brownbag/01:13:2022/att.csv" #with NA rows added to make dim consonant
att = pd.read_csv(path, header=0).to_numpy()
print(att)
print(np.shape(att))


# In[3]:


# showing combined plots!
tab_m = music[:,0]
D1_m = music[:,1]
D2_m = music[:,2]
tab_a = att[:,0]
D1_a = att[:,1]
D2_a = att[:,2]
print(tab_m)
fig, ax = plt.subplots()

plt.xticks(np.arange(-3, 3, 0.5))
plt.yticks(np.arange(-3, 3, 0.5))

plt.xlabel("D1")
plt.ylabel("D2")
plt.axis('scaled')
ax.grid(True, which='both')

plt.scatter(D1_m,D2_m,alpha=0.7,s=5)

  
# zip joins x and y coordinates in pairs
for t,x,y, a,b,c in zip(tab_m,D1_m,D2_m,tab_a,D1_a,D2_a):
    label_m = f"{t}\n({x},{y})"
    plt.annotate(label_m, # this is the text for music
                 (x, y), # this is the point to label
                 fontsize=4,
                 textcoords="offset points", # how to position the text
                 xytext=(0,2), # distance from text to points (x,y)
                 alpha=0.7, color='0.8',
                 ha='center',) # horizontal alignment can be left, right or center
    
    label_a = f"att_{a}\n({b:.3f},{c:.3f})"
    plt.annotate(label_a, # this is the text for att
                 (b, c), # this is the point to label
                 fontsize=6,
                 textcoords="offset points", # how to position the text
                 xytext=(0,2), # distance from text to points (x,y)
                 alpha=0.9, color='crimson',
                 ha='center') # horizontal alignment can be left, right or center

for i in range(np.shape(att)[0]):
    array = np.array([[0, 0, D1_a[i], D2_a[i]]])
    #print(f"{D1[i], D2[i]}")
    X, Y, U, V = zip(*array)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    ax = plt.gca()
    ax.quiver(X, Y, U, V,color='coral', angles='xy', scale_units='xy',scale=1)

plt.savefig('/Volumes/CHAIZHI/G25 PRD lab/brownbag/01:13:2022/att_D1vsD2_join.png', dpi=400)
plt.show()


# In[4]:


# showing separate plots
tab_m = music[:,0]
D1_m = music[:,1]
D2_m = music[:,2]

fig, ax = plt.subplots()

plt.scatter(D1_m,D2_m, s=8)
plt.xticks(np.arange(-3, 3, 0.5))
plt.yticks(np.arange(-3, 3, 0.5))
plt.xlabel("D1")
plt.ylabel("D2")
plt.axis('scaled')
ax.grid(True, which='both')

# zip joins x and y coordinates in pairs
for t,x,y in zip(tab_m,D1_m,D2_m):
    label = f"{t}\n({x},{y})"
    plt.annotate(label, # this is the text
                 (x, y), # this is the point to label
                 fontsize=6,
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 #alpha=overlapping,
                 ha='center',) # horizontal alignment can be left, right or center

plt.savefig('/Volumes/CHAIZHI/G25 PRD lab/brownbag/01:13:2022/D1 vs D2.png', dpi=200)


tab_a = att[:,0]
D1_a = att[:,1]
D2_a = att[:,2]
fig, ax = plt.subplots()
plt.xticks(np.arange(-3, 3, 0.5))
plt.yticks(np.arange(-3, 3, 0.5))
plt.xlabel("D1")
plt.ylabel("D2")
plt.axis('scaled')
ax.grid(True, which='both')

for i in range(np.shape(att)[0]):
    array = np.array([[0, 0, D1_a[i], D2_a[i]]])
    #print(f"{D1[i], D2[i]}")
    X, Y, U, V = zip(*array)
    plt.xlim(-2.5, 2)
    plt.ylim(-2.5, 2)
    ax = plt.gca()
    ax.quiver(X, Y, U, V,color='r', angles='xy', scale_units='xy',scale=1)

# zip joins x and y coordinates in pairs
for t,x,y in zip(tab_a,D1_a,D2_a):
    label = f"att_{t}\n({x:.3f},{y:.3f})"
    plt.annotate(label, # this is the text
                 (x, y), # this is the point to label
                 fontsize=4,
                 textcoords="offset points", # how to position the text
                 xytext=(0,2), # distance from text to points (x,y)
                 #alpha=overlapping,
                 ha='center') # horizontal alignment can be left, right or center
plt.savefig('/Volumes/CHAIZHI/G25 PRD lab/brownbag/01:13:2022/att_D1vsD2.png', dpi=200)
plt.show()

