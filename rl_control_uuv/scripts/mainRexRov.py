#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import rospy
import time
import sys
import os
import math

from models import ValueNetwork, SoftQNetwork, PolicyNetwork
from envRexRov import Env
from std_msgs.msg import Float32
from random import randint, randrange

from uuv_world_ros_plugins_msgs.srv import SetCurrentVelocity


# Use Cuda GPU, if not available CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Is CUDA available :"+str(torch.cuda.is_available())+"\n")
print(str(torch.version.cuda)+"\n")
#print(str(cudaDriverGetVersion())+"\n")

# --- Paths --- #
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
dirPath = os.path.dirname(os.path.realpath(__file__))
'''
# Topics that are used to send data to the superviser
cmd_learning_topic = rospy.Publisher("learning_cmd_vel", Twist, queue_size=1)
cmd_supervised_topic = rospy.Publisher("superviser_cmd_vel", Twist, queue_size=1)
'''
action_dim = 6      
state_dim  = 23  # 23 si on ne met que l'état actuel et pas les 4 états passés dans le vecteur d'état. Si on change le nombre d'état passés à rajouter dans le vecteur d'état, changer cette dimension et l'initialisation de self.obs dans la fonction reset deu fichier env
hidden_dim = 256    # Number of hidden nodes

ACTION_V_MIN = -240.    
ACTION_V_MAX = 240. 

''' Reward function parameters '''
lambda1 = 40.0
lambda2 = 0.0
lambda3 = 0.0

# Value and Soft_Q models have same criterion
# that measures the mean squared error (squared L2 norm)
# between each element in the input x and target y
value_criterion  = nn.MSELoss()
soft_q_criterion = nn.MSELoss()

# All models have same Learning Rate
# 3e-4 ou 1e-3
value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

# Training format
MAX_EPISODES  = 5001
MAX_STEPS     = 1000   # si on va au delà de 1000, penser à changer la définition des courants pour pas avoir l'erreur list index out of range
batch_size    = 256

reward_last_hundred_ep = []
rewards_all_episodes =   []

positive_collision_count = 0.
positive_collision_reward_mean = 0.

# Figures init
rewards_all_episodes = []

f, axarr = plt.subplots(4, sharex=False) # Subplot definition
f.tight_layout()

cpt   = 0   # Used for Potential-Based Reward Shaping

do_plot = False # Used to activate or not plotting of losses values
token   = True # Used to change environment while training

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.gamma = 0.99

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position+1) % self.capacity

        #print(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # batch = self.batch à la place pour prendre le batch rempli de PID
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Initialize buffer with size of 5.10^6
replay_buffer_size = 5000000
replay_buffer = ReplayBuffer(replay_buffer_size)

# Saving function
def save_models(episode_count,save_directory):
    torch.save(policy_net.state_dict(), dirPath + '/'+str(save_directory)+'/' + str(episode_count)+ '_policy_net.pth')
    torch.save(value_net.state_dict(), dirPath + '/'+str(save_directory)+'/' + str(episode_count)+ 'value_net.pth')
    torch.save(soft_q_net.state_dict(), dirPath + '/'+str(save_directory)+'/'+ str(episode_count)+ 'soft_q_net.pth')
    torch.save(target_value_net.state_dict(), dirPath + '/'+str(save_directory)+'/' + str(episode_count)+ 'target_value_net.pth')
    print("=================================self.===")
    print("Model has been saved...")
    print("=========================================")

def save_switch(ep, nb, token):

    if nb >= 90:
        save_models(90,save_directory)
    if nb >= 80:
        save_models(80,save_directory)
    if nb >= 70:
        save_models(70,save_directory)

    return token, ep

# Loading function
def load_models(episode, username, workspace_name, save_directory):
    policy_net.load_state_dict(torch.load('/home/'+str(username)+'/'+str(workspace_name)+'/src/rl_control_uuv/scripts/'+str(save_directory)+'/'+ str(episode)+ '_policy_net.pth'))
    value_net.load_state_dict(torch.load('/home/'+str(username)+'/'+str(workspace_name)+'/src/rl_control_uuv/scripts/'+str(save_directory)+'/' + str(episode)+ 'value_net.pth'))
    soft_q_net.load_state_dict(torch.load('/home/'+str(username)+'/'+str(workspace_name)+'/src/rl_control_uuv/scripts/'+str(save_directory)+'/'+ str(episode)+ 'soft_q_net.pth'))
    target_value_net.load_state_dict(torch.load('/home/'+str(username)+'/'+str(workspace_name)+'/src/rl_control_uuv/scripts/'+str(save_directory)+'/'+ str(episode) + 'target_value_net.pth'))
    print('***Models load***')

# Normalization function
def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action

def stats(reward_last_hundred_ep, rewards_all_episodes, total_Mean, total_success, nb,
          total_Mean_100, total_success_100, nb_step, username, workspace_name):
    mean = 0
    mean_100 = 0

    for i in reward_last_hundred_ep:
        mean_100 = mean_100 + i
    mean_100 = int(mean_100 / 100)

    for i in rewards_all_episodes:
        mean = mean + i
    mean = int(mean / ep)

    total_Mean.append(mean)
    total_success = total_success + nb
    total_Mean_100.append(mean_100)
    total_success_100.append(nb)

    print('Mean reward for last 100 ep : ' + str(mean_100))
    print('Total mean reward for each 100 ep : ' + str(total_Mean_100))
    print('Total mean reward : ' + str(total_Mean))
    print('Number of success for last 100 ep : ' + str(nb))
    print('Number of success for each 100 ep : ' + str(total_success_100))
    print('Total success : ' + str(total_success))
    print('Number of steps : ' + str(nb_step))
    print('Number of collisions where the total reward is positive : ' + str(positive_collision_count) + '  ,  mean total reward : ' + str(positive_collision_reward_mean))

    file2 = open("/home/"+str(username)+"/"+str(workspace_name)+"/src/rl_control_uuv/scripts/stats.txt", "a")
    file2.write("---------------------------------------------- \n")
    file2.write("Mean reward for last 100 ep : " + str(mean_100)+"\n")
    file2.write("Total mean reward for each 100 ep : " + str(total_Mean_100)+"\n")
    file2.write("Total mean reward : " + str(total_Mean)+"\n")
    file2.write("Number of success for last 100 ep : " + str(nb)+"\n")
    file2.write("Number of success for each 100 ep : " + str(total_success_100)+"\n")
    file2.write("Total success : " + str(total_success)+"\n")
    file2.write("Number of steps : " + str(nb_step)+"\n")
    file2.write("Number of collisions where the total reward is positive : " + str(positive_collision_count) + "  ,  mean total reward : " + str(positive_collision_reward_mean)+"\n")
    file2.close()
 
    reward_last_hundred_ep = []
    nb = 0

    return reward_last_hundred_ep, rewards_all_episodes, total_Mean, total_success, nb, \
           total_Mean_100, total_success_100, nb_step, nb, reward_last_hundred_ep

#----------------------------------------------------------#
#----------------------------------------------------------#
#-------------------- UPDATE FUNCTION ---------------------#
#                                                          #
# We we update the two Q function parameters by reducing   #
# the MSE between the predicted Q value for a state-action #
# pair and its corresponding.                              #
# For the V network update, we take the minimum of the two #
# Q values for a state-action pair and subtract from it the#
# Policy's log probability of selecting that action in that#
# state. Then we decrease the MSE between the above        #
# quantity and the predicted V value of that state.        #
# Then, we update the Policy parameters by reducing the    #
# Policy's log probability of choosing an action in a state#
# log(π(S)) minus the predicted Q-value of that            #
# state-action pair. Note here that in this loss, the      #
# predicted q value is composed of the policy : Q(S, π(S)).#
# This is important because it makes the term dependent on #
# the Policy parameters ϕ.                                 #
# Lastly, we update the Target Value Network by Polyak     #
# averaging it with the Main Value Network.                #
#                                                          #
#----------------------------------------------------------#
#----------------------------------------------------------#

def soft_q_update(batch_size, step, iter, rewards_all_episodes, ep, update,
           gamma=0.99,
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0,
           soft_tau=5e-3, #mettre à 1.0 pour la hard update, ou 5e-3 pour la soft update (1e-2 is also a good choice)
           total_q_value_loss=[],
           total_value_loss=[],
           total_policy_loss=[],
          ):

    # Get data from buffer
    state, action, reward, next_state, done = replay_buffer.sample(batch_size) # je charge le fichier text au lieu de sample
     

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    # apprentissage
    expected_q_value = soft_q_net(state, action)
    expected_value = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

    '''
    We train Q-Value Network by minimizing the following error :

        Jq = E[1/2(Q(st,at) - Qexp(st,at))^2]
    
    where Qexp = r(st,at) + gamma * E[V(st+1)]
        
    Gradient this is equal to :
    
        Grad(Jq) = Grad Q(at,st) (Q(st,at) - r(st,at) - gamma * V(st+1))
    '''

    target_value = target_value_net(next_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

    '''
    We train Value Network by minimizing the following error :
        
        Jv = E[1/2(V(st) - E[Q(st, at) - log π(at|st)])^2]
        
    Gradient thus is equal to :
    
        Grad(Jv) = Grad V(st)(V(st) - Q(st,at) + log π(at|st))
    '''

    expected_new_q_value = soft_q_net(state, new_action)
    next_value = expected_new_q_value - log_prob
    value_loss = value_criterion(expected_value, next_value.detach())

    '''
        We train Policy Network by minimizing the following error :

            Jπ = E[D_KL(π(.|st) || (exp(Q(st|.))) / (Z(st)))]
    
    -> We try to make the distribution of our Policy function to
       look more like the distribution of the exponentiation of
       our Q function normalized by another function Z().
    '''

    log_prob_target = expected_new_q_value - expected_value
    policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

    mean_loss = mean_lambda * mean.pow(2).mean()
    std_loss  = std_lambda  * log_std.pow(2).mean()
    z_loss    = z_lambda    * z.pow(2).sum(1).mean()

    policy_loss += mean_loss + std_loss + z_loss

    # Gradients Backpropagation

    soft_q_optimizer.zero_grad()
    q_value_loss.backward()
    soft_q_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()



    # PRINT LOSSES
    if step != 0 and step % 100 == 0 and update == True:
        print('Iter :- ', iter,
              ' Q_value Loss :- ', round(q_value_loss.data.item(), 2),
              ' Value Loss :- ', round(value_loss.data.item(), 2),
              ' Policy Loss :- ', round(policy_loss.data.item(), 2))


    # SOFT UPDATE
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    '''
    # HARD UPDATE
    if iter % 1000 == 0 and update == True:
	for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        	target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
	print("//////")
	print("Update")
	print("//////")
    '''


    # Store losses
    if step % 100 == 0:
        total_q_value_loss.append(q_value_loss.item())
        total_value_loss.append(value_loss.item())
        total_policy_loss.append(policy_loss.item())
    
    # Losses and Rewards plotting from text files
    if ep % 25 == 0 and do_plot == True:

        axarr[0].plot(rewards_all_episodes, color='c')
        axarr[0].set_title('Total reward per episode')

        axarr[1].plot(total_q_value_loss, color='r')
        axarr[1].set_title('Q_Value Loss')

        axarr[2].plot(total_value_loss, color='b')
        axarr[2].set_title('Value Loss')

        axarr[3].plot(total_policy_loss, color='g')
        axarr[3].set_title('Policy Loss')
        axarr[3].set_xlabel('Step iteration')

        plt.pause(0.000001)
    
    if ep % 500 == 0:
        f.savefig('/home/'+str(username)+'/'+str(workspace_name)+'/src/rl_control_uuv/scripts/'+str(save_directory)+'/' + str(ep) + '.png', bbox_inches='tight')

#----------------------------------------------------------#
#----------------------------------------------------------#
#----------------- MODELS INITIALIZATION ------------------#
#----------------------------------------------------------#
#----------------------------------------------------------#

value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

print('State Dimensions: ' + str(state_dim))
print('Action Dimensions: ' + str(action_dim))
print('Action Max: ' + str(ACTION_V_MAX))

username = "solayo"
workspace_name = "catkin_ws"
save_directory = "models"

is_training = True # If False, entering Testing mode
#load_models(5000, username, workspace_name, save_directory)
pid_buffer = False
pid_buffer_half_episodes = False # si on voit moitie episodes SAC et moitie episodes PID

#cmd = 'touch delta2_' + str(lambda2) + '_delta3_' + str(lambda3) + '.txt'
#file1 = open("touch delta2_' + str(delta2) + '_delta3_' + str(delta3) + '.txt","wb")

if __name__ == '__main__':
    rospy.init_node('sac')

    pub_result = rospy.Publisher('result', Float32, queue_size=5)

    start_time = time.time()
    result = Float32()
    env = Env()

    total_Mean_100 = []
    total_Mean = []

    total_success_100 = []
    total_success = 0

    nb_step = 0
    iter = 0
    phi = 0
    nb = 0
    results=[]
    past_action = np.array([0., 0., 0., 0., 0., 0.])
    seq = []

    nb_target = 1

    vitesse_max_x = 0.0
    vitesse_max_y = 0.0
    vitesse_max_z = 0.0

    received_cmd0_max = 0.0
    received_cmd1_max = 0.0
    received_cmd2_max = 0.0
    received_cmd3_max = 0.0
    received_cmd4_max = 0.0
    received_cmd5_max = 0.0
    received_cmd_overall_max = 0.0

    for ep in range(MAX_EPISODES):

        print('Episode: ' + str(ep))

        # mettre ep % 2 si on est en mode comparaison avec le pid, et 1 sinon
        if ep % 1 == 0:
            velocities = []
            angles_hori = []
            angles_verti = []
            for _ in itertools.repeat(None, 10):
                velocities.append(round(random.uniform(0., 1.),3))
                angles_hori.append(round(random.uniform(-0.5, .5), 3))
                angles_verti.append(round(random.uniform(-0.5, .5), 3))
            print(velocities)

        rospy.wait_for_service('/hydrodynamics/set_current_velocity')
        try:
            set_current_velocity_proxy = rospy.ServiceProxy('/hydrodynamics/set_current_velocity', SetCurrentVelocity)
            set_current_velocity = set_current_velocity_proxy(velocities[0], angles_hori[0], angles_verti[0])
            print('Current velocity: ' + str(velocities[0]) + ' Horizontal angle : ' + str(angles_hori[0]) + ' Vertical angle : ' + str(angles_verti[0]))
        except (rospy.ServiceException) as e:
            print("/hydrodynamics/set_current_velocity call failed")

        '''
        velocity = round(random.uniform(0, 1.), 3)
        angle_hori = round(random.uniform(-0.5, 0.5), 3)
        angle_verti = round(random.uniform(-0.5, 0.5), 3)
        current = [velocity, angle_hori, angle_verti]
        cmd = ' rosservice call /hydrodynamics/set_current_velocity "{velocity: ' + str(
            velocity) + ',  horizontal_angle: ' + str(angle_hori) + ', vertical_angle: ' + str(angle_verti) + '}"'
        os.system(cmd)
        print('Current velocity: ' + str(velocity) + ' Horizontal angle : ' + str(angle_hori) + ' Vertical angle : ' + str(angle_verti))
        switch_current = randint(350,750)
        '''
        done = False

        state_full, state, past_state, goal_x, goal_y, goal_z, position_x, position_y, position_z, pid_mode = env.reset(token, 1, ep)

        rewards_current_episode = 0
        token = False
        cumul_reward = []
        target_number = 0

        ind = 0




        # si on voit moitie episodes SAC et moitie episodes PID
        if pid_buffer_half_episodes:
            if( ep % 2 == 0 ):
                pid_buffer = False
                print("pid_buffer = False")
            else:
                pid_buffer = True
                print("pid_buffer = True")





        for step in range(MAX_STEPS):

            if step % 101 == 0 and step > 0:
		ind += 1

                '''
                cmd = ' rosservice call /hydrodynamics/set_current_velocity "{velocity: ' + str(
                    velocities[ind]) + ',  horizontal_angle: ' + str(angles_hori[ind]) + ', vertical_angle: ' + str(
                    angles_verti[ind]) + '}"'
                os.system(cmd)
                '''

                
                rospy.wait_for_service('/hydrodynamics/set_current_velocity')
                try:
                    set_current_velocity_proxy = rospy.ServiceProxy('/hydrodynamics/set_current_velocity', SetCurrentVelocity)
                    set_current_velocity = set_current_velocity_proxy(velocities[ind], angles_hori[ind], angles_verti[ind])
                    print('Current velocity: ' + str(velocities[ind]) + ' Horizontal angle : ' + str(angles_hori[ind]) + ' Vertical angle : ' + str(angles_verti[ind]))
                except (rospy.ServiceException) as e:
                    print("/hydrodynamics/set_current_velocity call failed")
                

                
            
            state = np.float32(state)
            nb_step += 1


            policy_net.eval() # on le met en mode evaluation, sinon une erreur est leve car batch_norm ne fonctionne quand le batch est de size 1 comme ici
            action = policy_net.get_action(state)
            policy_net.train() # on le met en mode training

            for i in range(len(action)) :
                 action[i] = action[i] + np.random.normal(0, random.uniform(0.01, .05), 1)


            unnorm_action = np.array([action_unnormalized(action[0], ACTION_V_MAX, ACTION_V_MIN), action_unnormalized(action[1], ACTION_V_MAX, ACTION_V_MIN),
                                      action_unnormalized(action[2], ACTION_V_MAX, ACTION_V_MIN), action_unnormalized(action[3], ACTION_V_MAX, ACTION_V_MIN),
                                      action_unnormalized(action[4], ACTION_V_MAX, ACTION_V_MIN), action_unnormalized(action[5], ACTION_V_MAX, ACTION_V_MIN)])


            #print("_______________________________")
            #print(step)
            #print(unnorm_action)
            #print(action)
            #print(past_action) # entre -1 et 1



            next_state_full, next_state, reward, done, reached, nb_success, reward_dist, reward_actuators, reward_lambda2, reward_lambda3, past_state, goal_x, goal_y, goal_z, position_x, position_y, position_z, received_cmd0, received_cmd1, received_cmd2, received_cmd3, received_cmd4, received_cmd5 = env.step(unnorm_action, past_action, past_state, step, lambda1, lambda2, lambda3, ep, pid_buffer)


            if(pid_buffer):
                action[0]=received_cmd0.data/ACTION_V_MAX
                action[1]=received_cmd1.data/ACTION_V_MAX
                action[2]=received_cmd2.data/ACTION_V_MAX
                action[3]=received_cmd3.data/ACTION_V_MAX
                action[4]=received_cmd4.data/ACTION_V_MAX
                action[5]=received_cmd5.data/ACTION_V_MAX

            #print(unnorm_action)
            #print(action)


            linear_speed_x = next_state_full[6]
            linear_speed_y = next_state_full[7]
            linear_speed_z = next_state_full[8]

            if np.abs(linear_speed_x) > vitesse_max_x:
                vitesse_max_x = np.abs(linear_speed_x)

            if np.abs(linear_speed_y) > vitesse_max_y:
                vitesse_max_y = np.abs(linear_speed_y)

            if np.abs(linear_speed_z) > vitesse_max_z:
                vitesse_max_z = np.abs(linear_speed_z)



            
            if np.abs(received_cmd0.data) > received_cmd0_max:
                received_cmd0_max = np.abs(received_cmd0.data)

            if np.abs(received_cmd1.data) > received_cmd1_max:
                received_cmd1_max = np.abs(received_cmd1.data)

            if np.abs(received_cmd2.data) > received_cmd2_max:
                received_cmd2_max = np.abs(received_cmd2.data)

            if np.abs(received_cmd3.data) > received_cmd3_max:
                received_cmd3_max = np.abs(received_cmd3.data)

            if np.abs(received_cmd4.data) > received_cmd4_max:
                received_cmd4_max = np.abs(received_cmd4.data)

            if np.abs(received_cmd5.data) > received_cmd5_max:
                received_cmd5_max = np.abs(received_cmd5.data)
            



            x_error = next_state_full[-3]
            y_error = next_state_full[-2]
            z_error = next_state_full[-1]
            current_distance = round(math.sqrt(x_error ** 2 + y_error ** 2 + z_error ** 2), 2)

            robot_init_position_vector = np.array([0.0,0.0,-20.0])
            goal_init_position_vector = np.array([goal_x,goal_y,goal_z])
            init_distance_vector = goal_init_position_vector - robot_init_position_vector
            current_distance_vector = np.array([x_error,y_error,z_error])

            robot_init_position_norm = np.linalg.norm(init_distance_vector)
            current_distance_norm = np.linalg.norm(current_distance_vector)

            # deviation from the ideal trajectory
            trajectory_error = current_distance_norm * np.sin( np.arccos( np.dot(init_distance_vector,current_distance_vector)/(robot_init_position_norm * current_distance_norm) ) )

            file1 = open("/home/"+str(username)+"/"+str(workspace_name)+"/src/rl_control_uuv/scripts/data.txt", "a")
            file1.write("%i, %i, %5.3f, %5.3f, %5.3f, " % (ep, int(pid_mode == True), velocities[ind], angles_hori[ind], angles_verti[ind]))
            file1.write("%3.3f, %3.3f, %3.3f, %3.3f, %3.3f, %3.3f, " % (received_cmd0.data, received_cmd1.data, received_cmd2.data, received_cmd3.data, received_cmd4.data, received_cmd5.data))
            #for element in next_state_full[98:]: # si l'état est contistué des 5 états précédents
            for element in next_state_full[6:]: 
                file1.write("%5.3f, " % element)
            file1.write("%5.3f, %i, %i, %i, " % (reward, int(done == True), int(reached == True), nb_success))
            file1.write("%5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %5.3f, %i\n" % (
            goal_x, goal_y, goal_z, position_x, position_y, position_z, trajectory_error, current_distance, step))
            file1.close()

            if (step >= 0) and (step % 50 == 0):
                print("Step number : " + str(step) + "/" + str(MAX_STEPS) + " ; Reward : " + str(reward) + " ; Dist : " + str(current_distance) + " ; reward_dist : "+ str(reward_dist)+ " ; reward_actuators : "+ str(reward_actuators) + " ; reward_lambda2 : "+ str(reward_lambda2) + " ; reward_lambda3 : "+ str(reward_lambda3) +" ; Target number : " + str(target_number+1) + " ; vitesse_max_x = "+str(vitesse_max_x) + " ; vitesse_max_y = "+str(vitesse_max_y) + " ; vitesse_max_z = "+str(vitesse_max_z)+ " ; received_cmd0_max = "+str(received_cmd0_max)+ " ; received_cmd1_max = "+str(received_cmd1_max)+ " ; received_cmd2_max = "+str(received_cmd2_max)+ " ; received_cmd3_max = "+str(received_cmd3_max)+ " ; received_cmd4_max = "+str(received_cmd4_max)+ " ; received_cmd5_max = "+str(received_cmd5_max)  )

            past_action = action
            rewards_current_episode += reward

            next_state = np.float32(next_state)

            #print(state)
            #print(reward)

            replay_buffer.push(state, action, reward, next_state, done)


            '''
            print("_______________________________")
            print(step)
            #print("%3.3f, %3.3f, %3.3f, %3.3f, %3.3f, %3.3f, " % (received_cmd0.data, received_cmd1.data, received_cmd2.data, received_cmd3.data, received_cmd4.data, received_cmd5.data))
            #print(action)
            #print(unnorm_action)
            #print(reward)
            #print(state)
            print("-----------------------------")
            '''



            if step == 50:
                do_plot = True
            else:
                do_plot = False

            # Training start after buffe size > 2 * MAX_STEPS
            if len(replay_buffer) >= 2 * MAX_STEPS and is_training:
                #soft update : que la 1ère ligne, commenter les 3 autres ; hard update : utilise les 4 lignes
                soft_q_update(batch_size, step, iter, rewards_all_episodes, ep, update = True)
                #soft_q_update(batch_size, step, iter, rewards_all_episodes, ep, update = False) 
                #soft_q_update(batch_size, step, iter, rewards_all_episodes, ep, update = False)
                #soft_q_update(batch_size, step, iter, rewards_all_episodes, ep, update = False)

                iter += 1
                cpt  += 1

            state = next_state

            if reached:
                target_number += 1

                if reached and target_number < nb_target:
                    env.set_next_target()
                reached = False

            # End of current episode
            if (done) or (step == MAX_STEPS - 1) or (target_number > nb_target) and step >=1:

                env.standStill()  # Keep the robot immobile to avoid models merging

                if len(replay_buffer) >= 2 * MAX_STEPS and is_training:  # Start stocking rewards when models update has started
                    rewards_all_episodes.append(rewards_current_episode)

                reward_last_hundred_ep.append(rewards_current_episode)
                result = rewards_current_episode
                pub_result.publish(result)
                nb = nb + nb_success

                print('Total reward : ' + str(rewards_current_episode))

                target_number = 0

                if done and rewards_current_episode > 0:
                    positive_collision_count = positive_collision_count + 1.
                    positive_collision_reward_mean = positive_collision_reward_mean + (
                                rewards_current_episode - positive_collision_reward_mean) / positive_collision_count
                '''
                if ep % 100 == 0 and ep > 0:
                    token, ep = save_switch(ep, nb, token)
                '''
                if ep % 100 == 0 and ep > 0:
                    reward_last_hundred_ep, rewards_all_episodes, total_Mean, total_success, nb, \
                    total_Mean_100, total_success_100, nb_step, nb, reward_last_hundred_ep = \
                        stats(reward_last_hundred_ep, rewards_all_episodes,
                              total_Mean, total_success, nb, total_Mean_100, total_success_100, nb_step, username, workspace_name)
                break
        
        if is_training and ep < 1500 and ep % 100 == 0:
            save_models(ep,save_directory)

        if is_training and ep > 1500 and ep % 250 == 0:
            save_models(ep,save_directory)
        
    file1.close()

if is_training:
    print ('Training completed !!')
else:
    print ('Testing completed !!')
