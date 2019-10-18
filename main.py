import numpy as np
import random
import matplotlib.pyplot as plt
import math
import tensorflow as tf

def positionGeneration():
    d = 50 # diameter of the robot
    # Randomly generating theta (the angle between local reference frame of the robot and the global frame) 
    # and the left, right sensor positions
    theta = np.random.randint(0,360)
    x_l = np.random.randint(1+d,1000-d)
    y_l = np.random.randint(1+d,1000-d)
    x_r = int(x_l + math.cos(90-theta) * d)
    y_r = int(y_l - math.sin(90-theta) * d)
    
    return (np.array([x_l,x_r,y_l,y_r]))
    
def zvalue(position):
    # Creating z-matrix - shape(1000,1000) and filled from 1 (row[0]) to 1000 (row[999])
    x = np.arange(1,1000,1)
    y = np.arange(1,1000,1)
    xx, yy = np.meshgrid(x,y, sparse = True)
    z = np.tile(yy, (1, 999))
    # Finding the corresponding z value (light intensity) to the given position values
    z_l = z[int(position[0]), int(position[2])]
    z_r = z[int(position[1]), int(position[3])]

    return np.array([z_l, z_r])

def step(position, action):
    # taking action from the current state (current position)
    if action == 0:
        turn = 1
    else:
        turn = -1
    # Generating Rotation matrix using the given turn value
    c, s = np.cos(turn), np.sin(turn)
    R = np.array(((c, -s), (s, c)))
    x_l, x_r, y_l, y_r = position[0], position[1], position[2], position[3]
    x_c = (x_l + x_r)/2
    y_c = (y_l + y_r)/2
    # Turning by degree of 'turn'
    l_temp= np.array([x_l-x_c, y_l-y_c])
    l_new = np.matmul(l_temp, R) + np.array([x_c, y_c])
    r_temp= np.array([x_r-x_c, y_r-y_c])
    r_new = np.matmul(r_temp, R) + np.array([x_c, y_c])
    # Assinging new position value
    new_position = np.array([l_new[0], r_new[0], l_new[1], r_new[1]])
    
    return(new_position)


def rewardDone(state):
    if (abs(state[0] - state[1]) < 5):
        reward = 1
        done = True
    else:
        reward = 0
        done = False
    
    return(reward, done)

# Setting the neural network 
learning_rate = 1e-1       
input_size = 2
output_size = 1

action_space = np.array([0, 1])

X = tf.placeholder( tf.float32, [None, input_size], name = 'input_x')
W1 = tf.get_variable("W1", shape = [input_size, output_size], initializer = tf.contrib.layers.xavier_initializer())
Qpred = tf.matmul(X, W1)

Y = tf.placeholder( tf.float32, [None, output_size])

loss = tf.reduce_sum(tf.square(Y-Qpred))

train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# Hyperparameters
num_episodes = 2000
dis = 0.99
rList = []

# Q-learning
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for i in range( num_episodes):
    e = 1./((i+1)/10) # e-greedy value
    rAll = 0
    done = False
    step_count = 0
    position = positionGeneration()
    state = zvalue(position)
    initial_z_diff = abs(state[0] - state[1])

    while not done:
        step_count += 1
        x = np.reshape(state, [1, input_size]) # pre-processing the input data
        # Choose an action by greedily
        Qs = sess.run(Qpred, feed_dict={X: x})
        if np.random.rand(1) < e:
            action = np.random.randint(1, size = 0)
        else:
            action = np.argmax(Qs)

        # Result from the action
        new_position = step(position, action)
        new_state = zvalue(new_position)
        reward, done = rewardDone(new_state)

        # Updating Q-network
        x1 = np.reshape(new_state, [1, input_size])
        Qs1 = sess.run(Qpred, feed_dict={X: x1})
        Qs[0, action] = reward + dis * np.max(Qs1)

        sess.run(train, feed_dict = {X: x, Y: Qs})
        position = new_position
        state = new_state

    rList.append(step_count)
    print("Episodes: {}, steps: {}, initial difference between two wheels: {}".format(i, step_count, initial_z_diff))

    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break


# Actual Play!
position = positionGeneration()
observation = zvalue(position)
reward_sum = 0
while True:
    x = np.reshape(observation, [1, input_size])
    Qs = sess.run(Qpred, feed_dict={X: x})
    a = np.argmax(Qs)

    new_position = step(position, action)
    new_state = zvalue(new_position)
    reward, done = rewardDone(new_state)
    reward_sum += reward
    if done:
         print("Total score: {}".format(reward_sum))
         break



