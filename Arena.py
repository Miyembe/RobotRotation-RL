import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import dqn
import tensorflow as tf

# Define range of x and y for the arena
x = np.arange(1,1000,1)
y = np.arange(1,1000,1)
# Using meshgrid, xx and yy represent the coordinates
xx, yy = np.meshgrid(x,y, sparse = True)
# Making 2D array by concatenating yy horizontally (999x1) -> (999x999)
z = np.tile(yy, (1, 999))

d = 50 # diameter (mm)

dis = 0.6
REPLAY_MEMORY = 50000

Q = np.zeros((1000,1000,1000,1000,2), dtype = np.int8) # I don't know how to make a table. (I dont know how to make the shape of a table accordingly)
num_episodes = 2000

rList = []
for i in range(num_episodes):
    
    theta = np.random.randint(0,360)
    # position of the left sensor of the robot (randomly selected)
    x_l = np.random.randint(1+d,1000-d)
    y_l = np.random.randint(1+d,1000-d)
    # position of the right sensor of the robot 
    x_r = int(x_l + math.cos(90-theta) * d)
    y_r = int(y_l - math.sin(90-theta) * d)
    x_c = (x_l + x_r)/2
    y_c = (y_l + y_r)/2
    state = np.array([x_l,y_l,x_r,y_r], dtype = int)
    rAll = 0
    done = False
    while not done:
        action = np.argmax(Q[state[0], state[1], state[2], state[3], :]) # How can I set the action as turning theta +1 or -1
        #I have to make this new state in a more simple way. Too complicated
        if action == 0: # turning right
            turn = 1
        else:
            turn = -1
        c, s = np.cos(turn), np.sin(turn)
        R = np.array(((c, -s), (s, c)))
        l_temp= np.array([x_l-x_c, y_l-y_c])
        l_new = np.matmul(l_temp, R) + np.array([x_c, y_c])
        r_temp= np.array([x_r-x_c, y_r-y_c])
        r_new = np.matmul(r_temp, R) + np.array([x_c, y_c])
        new_state = np.array([int(l_new[0]), int(l_new[1]), int(r_new[0]), int(r_new[1])])
        # Make this as a function
        # I have to make a step function too.
        if (abs(z[x_l,y_l] - z[x_r,y_r]) < 5):
            reward = 1
            done = True
        else:
            reward = 0
        Q[state[0], state[1], state[2], state[3], action] = \
            reward + np.max(Q[new_state[0], new_state[1], new_state[2], new_state[3],:])
        theta += turn
        rAll += reward
        state = new_state
    rList.append(rAll)

print("Success Rate: " + str(sum(rList)/num_episodes))
plt.bar(range(len(rList)), rList, color="blue")
#print("the two sensor of the robots are at : ({},{}) and ({},{})", x_l, y_l, x_r, y_r)
# plotting z values in contour plot
#h = plt.contour(x,y,z, levels = 1000)
#plt.colorbar()
plt.show()

