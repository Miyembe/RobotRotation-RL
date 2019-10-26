import dqn
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque


# Input size and output size of the neural network
input_size = 2
output_size = 2

# Memory size for experience replay 
REPLAY_MEMORY = 50000

# Diameter of robot
d = 80

# Action space - only two elements (indicate rotate clockwise or anti-clockwise)
action_space = np.array([0, 1])

# Creating the z matrix - represent the arena of 1000 x 1000 - z value is assigned in gradient of 1 - 1000 (x: 1 - 1000, z: 1 - 1000)
x = np.arange(1,1000,1)
y = np.arange(1,1000,1)
xx, yy = np.meshgrid(x,y, sparse = True)
z = np.tile(yy, (1, 999))


def positionGeneration(d): # d =  diameter of the robot
    # Randomly generating theta (the angle between local reference frame of the robot and the global frame) 
    # and the left, right sensor positions
    theta = np.random.randint(0,360) # Theta is 0 when it faces the direction at which the maximum intensity is. 
    x_l = 500
    y_l = 500
    x_r = int(x_l + math.cos(90-theta) * d)
    y_r = int(y_l - math.sin(90-theta) * d)
    
    return (np.array([x_l,x_r,y_l,y_r]), theta)
    
def zvalue(position, z):
    # Finding the corresponding z value (light intensity) for the given position values
    z_l = z[int(position[0]), int(position[2])]
    z_r = z[int(position[1]), int(position[3])]

    return np.array([z_l, z_r])

def step(position, action):
    # taking action from the current state (current position)
    if action == 0:
        turn = math.pi / 180
    else:
        turn = - (math.pi) / 180
    # Generating Rotation matrix using the given turn value
    c, s = np.cos(turn), np.sin(turn)
    R = np.array(((c, s), (-s, c)))
    x_l, x_r, y_l, y_r = position[0], position[1], position[2], position[3]
    x_c = (x_l + x_r)/2
    y_c = (y_l + y_r)/2
    # Turning by degree of 'turn'
    l_temp= np.array([x_l-x_c, y_l-y_c])
    l_new = np.matmul(l_temp, R) + np.array([x_c, y_c])
    r_temp= np.array([x_r-x_c, y_r-y_c])
    r_new = np.matmul(r_temp, R) + np.array([x_c, y_c])
    # Assinging new position value
    next_position = np.array([l_new[0], r_new[0], l_new[1], r_new[1]])
    
    return(next_position)


def rewardDone(state, position):
    if (abs(state[0] - state[1]) < 10 and position[2] > position[3]):
        reward = 100
        done = True
    else:
        reward = 0
        done = False
    
    return(reward, done)

def replay_train(mainDQN, targetDQN, train_batch, dis):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    # Get stored information from the buffer

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # get target from target DQN (Q')
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    # Return mainDQN
    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    # Copy variables src_scope to dest_scope

    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def bot_play(mainDQN):

    position = positionGeneration(d)
    state = zvalue(position, z)
    reward_sum = 0
    while True:
        #env.render()
        action = np.argmax(mainDQN.predict(state))
        next_position = step(position, action)
        next_state = zvalue(next_position, z)
        reward, done = rewardDone(next_state)
        reward_sum += reward
        if done:
            print("Total score: {}".format((reward_sum)))
            break
def main():
    max_episodes = 3000
    rList = []
    dis = 0.8
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        # initial copy q_net = target_net
        copy_ops = get_copy_var_ops(dest_scope_name = "target", src_scope_name = "main")
        # Copying from target to main    
        sess.run(copy_ops)

        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            position, initial_angle_diff = positionGeneration(d)
            state = zvalue(position, z)
            while not done:
                if np.random.rand(1) < e:
                    action = np.random.choice(action_space)
                else:
                    # action from mainDQN
                    action = np.argmax(mainDQN.predict(state))

                # state update

                next_position = step(position, action)
                next_state = zvalue(next_position, z)
                reward, done = rewardDone(next_state, next_position)
                #print("current position: {}, next position: {}".format(position, next_position))
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) == REPLAY_MEMORY:
                    replay_buffer.popleft()

                position = next_position
                state = next_state
                step_count += 1
                print("step: {}, action: {}, state: {} ".format(step_count, action, state))
                #if step_count > 30000:
                #    break
            if (initial_angle_diff > 1 and initial_angle_diff < 359):
                if (initial_angle_diff < 180):  
                    rList.append(step_count/initial_angle_diff)
                else:
                    rList.append(step_count/( 360 - initial_angle_diff))
            else:
                rList.append(0)

            if initial_angle_diff <= 180 :
                print("Episodes: {}, steps: {}, initial angle difference: {}".format(episode, step_count, initial_angle_diff))
            else:
                print("Episodes: {}, steps: {}, initial angle difference: {}".format(episode, step_count, - (360 - initial_angle_diff)))
            
            
            if episode % 5 == 1:
                    for _ in range(50):
                        minibatch = random.sample(replay_buffer, 5)
                        loss, _ = replay_train(mainDQN, targetDQN, minibatch, dis)
                    print("Loss: ", loss)
    
    plt.bar(range(len(rList)), rList, color="blue")
    plt.show()
        

if __name__ == "__main__":
    main()