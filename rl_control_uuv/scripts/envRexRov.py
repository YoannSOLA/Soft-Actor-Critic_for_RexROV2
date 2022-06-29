import math
import rospy
import gazebo_env
import random
import numpy as np
import os

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist, Point, Quaternion
from gazebo_msgs.msg import ContactsState, ModelState
from gazebo_msgs.srv import SetModelState
from multiStageRespawn import Respawn
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from sensor_msgs.msg import FluidPressure
from std_srvs.srv import Empty
from cv_bridge import CvBridge
from random import randint
from math import pi, sqrt

from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from uuv_control_msgs.srv import *

from uuv_thrusters.models import Thruster
import tf2_ros
from tf.transformations import quaternion_matrix

from uuv_control_msgs.srv import GoTo
from uuv_control_msgs.msg import Waypoint
from std_msgs.msg import Time

'''
    Our state vector is compose of 3 frames from the Depth camera, the relative distance
    and orientation of the robot toward the target and the past actions realized.
    
    1 frame = 10 depths values
    Relative distance = 1 value (cm)
    Relative orientation = 1 value (degrees)
    Past actions = 1 Velocity value (m/s) and 1 Angular value (rad/s)
    
    State Vector = [Frame(t) + Frame(t-1) + Frame(t-2) + Distance(t) + Orientation(t) + Velocity(t-1) + Angular(t-1)]
    
    Size of State Vector = 34
'''

class Env(gazebo_env.GazeboEnv):

    def __init__(self):

        self.position = Point()
        self.linear_speed = Point()
        self.angular_speed = Point()

        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        self.pitch_error = 0
        self.yaw_error = 0

        self.goal_x = 0
        self.goal_y = 0
        self.goal_z = 0

        self.nb_success = 0
        self.past_distance = 0
        self.initGoal = True
        
        self.mean_actions = [0.,0.,0.,0.,0.,0.]
        self.mean_actions_length = 100.
        self.actions_list = []

        self.respawn_goal = Respawn()

        self.pub_cmd_vel0 = rospy.Publisher('/rexrov2/thrusters/0/input', FloatStamped, queue_size=1)
        self.pub_cmd_vel1 = rospy.Publisher('/rexrov2/thrusters/1/input', FloatStamped, queue_size=1)
        self.pub_cmd_vel2 = rospy.Publisher('/rexrov2/thrusters/2/input', FloatStamped, queue_size=1)
        self.pub_cmd_vel3 = rospy.Publisher('/rexrov2/thrusters/3/input', FloatStamped, queue_size=1)
        self.pub_cmd_vel4 = rospy.Publisher('/rexrov2/thrusters/4/input', FloatStamped, queue_size=1)
        self.pub_cmd_vel5 = rospy.Publisher('/rexrov2/thrusters/5/input', FloatStamped, queue_size=1)

        self.cmd1 = FloatStamped()
        self.cmd2 = FloatStamped()
        self.cmd3 = FloatStamped()
        self.cmd4 = FloatStamped()
        self.cmd5 = FloatStamped()
        self.cmd6 = FloatStamped()


        self.sub_cmd_vel0 = rospy.Subscriber('/rexrov2/thrusters/0/input', FloatStamped, self.cmd_vel0_callback)
        self.sub_cmd_vel1 = rospy.Subscriber('/rexrov2/thrusters/1/input', FloatStamped, self.cmd_vel1_callback)
        self.sub_cmd_vel2 = rospy.Subscriber('/rexrov2/thrusters/2/input', FloatStamped, self.cmd_vel2_callback)
        self.sub_cmd_vel3 = rospy.Subscriber('/rexrov2/thrusters/3/input', FloatStamped, self.cmd_vel3_callback)
        self.sub_cmd_vel4 = rospy.Subscriber('/rexrov2/thrusters/4/input', FloatStamped, self.cmd_vel4_callback)
        self.sub_cmd_vel5 = rospy.Subscriber('/rexrov2/thrusters/5/input', FloatStamped, self.cmd_vel5_callback)

        self.received_cmd0 = FloatStamped()
        self.received_cmd1 = FloatStamped()
        self.received_cmd2 = FloatStamped()
        self.received_cmd3 = FloatStamped()
        self.received_cmd4 = FloatStamped()
        self.received_cmd5 = FloatStamped()

        self.sub_odom = rospy.Subscriber('/rexrov2/pose_gt', Odometry, self.getOdometry)

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        self.reset_controller_proxy = rospy.ServiceProxy('/rexrov2/reset_controller', ResetController)
        self.set_pid_mode_proxy = rospy.ServiceProxy('/rexrov2/set_pid_mode', SetPIDMode)

        self.namespace = rospy.get_namespace().replace('/', '')
        self.base_link = rospy.get_param('~base_link', 'base_link')

        self.comparison_mode = rospy.get_param('~comparison_mode', 'false')
        self.j = 0
        self.pid_mode = False

        self.full_pid_mode = False

        self.time1 = 0.0
        self.time2 = 0.0

        self.vitesse_max_x = 0.0
        self.vitesse_max_y = 0.0
        self.vitesse_max_z = 0.0

    # Compute euclidean distance between robot and target
    def getGoalDistace(self):
        #goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        goal_distance = round(math.sqrt((self.goal_x - self.position.x)**2 + (self.goal_y - self.position.y)**2 + (self.goal_z - self.position.z)**2), 2)
        self.past_distance = goal_distance

        return goal_distance

    def cmd_vel0_callback(self, cmd):
        self.received_cmd0=cmd

    def cmd_vel1_callback(self, cmd):
        self.received_cmd1=cmd

    def cmd_vel2_callback(self, cmd):
        self.received_cmd2=cmd

    def cmd_vel3_callback(self, cmd):
        self.received_cmd3=cmd

    def cmd_vel4_callback(self, cmd):
        self.received_cmd4=cmd

    def cmd_vel5_callback(self, cmd):
        self.received_cmd5=cmd


    # Compute orientation of the robot toward the target
    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)

        goal_yaw = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        yaw_error = goal_yaw - yaw

        if yaw_error > pi:
            yaw_error -= 2 * pi

        elif yaw_error < -pi:
            yaw_error += 2 * pi

        goal_distance = round(math.sqrt((self.goal_x - self.position.x)**2 + (self.goal_y - self.position.y)**2 + (self.goal_z - self.position.z)**2), 2)
        #goal_pitch = math.atan2(self.goal_x - self.position.x, self.goal_z - self.position.z)
        goal_pitch = -math.atan2(self.goal_z - self.position.z, goal_distance)
        pitch_error = goal_pitch - pitch

        if pitch_error > pi:
            pitch_error -= 2 * pi

        elif pitch_error < -pi:
            pitch_error += 2 * pi
        
        self.yaw_error = round(yaw_error,3)
        self.pitch_error = round(pitch_error,3)

        self.yaw = round(yaw,3)
        self.pitch = round(pitch, 3)
        self.roll = round(roll,3)

        self.linear_speed = odom.twist.twist.linear
        self.angular_speed = odom.twist.twist.angular

    def noise(self, noisefree_data):

        data_w_noise = noisefree_data + np.random.normal(0, random.uniform(.1, .05), 1)

        return data_w_noise

    # Generate state
    def getState(self, data1, past_action, past_state, ite, ep):

        state = past_state


        state.pop()
        state.pop()
        state.pop()
        state.pop()
        state.pop()
        state.pop()


        yaw = self.yaw
        pitch = self.pitch
        roll = self.roll
        done = False
        reached = False
        #bridge = CvBridge()
        #cv_image = bridge.imgmsg_to_cv2(data)  # Transform depth/image_raw into depth map
        '''
        # Contact detector based on the ROS bumper plugin
        contact_data = None
        while contact_data is None:
            contact_data = rospy.wait_for_message('/contact_state', ContactsState, timeout=5) # To use this pluging you need to attach the bumper
        collision = contact_data.states != []                                                 # to a link of your model in the urdf.xacro description file
                                                                                              # Check the actual link's name of your robot in Gazebo sim.
        if collision:
            done = True
        
        for i in xrange(0, 100): # Keep 100 values from an horizontal line from depth map
            a = (680 / 100) * i - 1
            depth.append(float(cv_image[250, a]))
        '''

        '''
        if (self.position.z > -1.) or (self.position.z < -60.) or ( np.abs(self.position.x) > 30.0) or ( np.abs(self.position.y) > 30.0):
            if ite > 1:
                done = True
        else:
            done = False
        '''

        if (self.position.z > -1.) or (self.position.z < -60.):
            if ite > 1:
                done = True
        else:
            done = False

        past_state = state
        '''
        # Uncomment to have State vector = [ Frame(t) + frame (t-1) + Frame(t)-Frame(t-1) + Past actions + Heading + Distance ] 
        for i in xrange(0, 10):
            state.insert(20, round((state[i]-state[i+10]), 5))
            state.pop()
        '''
        # Reach target detector
        current_distance = round(math.sqrt((self.goal_x - self.position.x)**2 + (self.goal_y - self.position.y)**2 + (self.goal_z - self.position.z)**2),2)

        '''
	if (self.position.z <= 0.25) or (self.position.z >= 12.0) or ( abs(self.position.x) >= 20.0) or ( abs(self.position.y) >= 20.0):
            if int(step_nb) > 1:
                done = True
	else:
            done = False

        '''

        '''
        if ep <= 300 and current_distance < 25.0:
            reached = True
            print("ep="+str(ep)+" <= 300 AND current_distance="+str(current_distance)+" < 25.0\n")

        if 300 < ep and ep <= 600 and current_distance < 10.0:
            reached = True
            print("300 < ep="+str(ep)+" <= 600 AND current_distance="+str(current_distance)+" < 10.0\n")

        if 600 < ep and current_distance < 3.0:
            reached = True
            print("600 < ep="+str(ep)+" AND current_distance="+str(current_distance)+" < 3.0\n")
        '''



        if current_distance < 3.0:
            reached = True

        # Add past actions to state
        for pa in past_action:
            pa_normalized = (pa+1.0)/2.  # entre 0 et 1
            state.append(round(pa_normalized,3))

        self.position.x = round(self.position.x,3)
        self.position.y = round(self.position.y,3)
        self.position.z = round(self.position.z,3)

        self.linear_speed.x = round(self.linear_speed.x,3)
        self.linear_speed.y = round(self.linear_speed.y,3)
        self.linear_speed.z = round(self.linear_speed.z,3)

        self.angular_speed.x = round(self.angular_speed.x,3)
        self.angular_speed.y = round(self.angular_speed.y,3)
        self.angular_speed.z = round(self.angular_speed.z,3)

        x_error = round((self.goal_x - self.position.x),3)
        y_error = round((self.goal_y - self.position.y),3)
        z_error = round((self.goal_z - self.position.z),3)

        #print(current_distance)
        '''
        print state + [self.linear_speed.x, self.linear_speed.y, self.linear_speed.z,
            self.angular_speed.x, self.angular_speed.y, self.angular_speed.z,
            roll, pitch, yaw, x_error, y_error, z_error]
        '''



        # -------------------  TEST DE VECTEURS D'ETAT  -------------------

        state_full = state + [round(self.noise(self.linear_speed.x),3), round(self.noise(self.linear_speed.y),3), round(self.noise(self.linear_speed.z),3),
            round(self.noise(self.angular_speed.x),3), round(self.noise(self.angular_speed.y),3), round(self.noise(self.angular_speed.z),3),
            self.noise(roll), self.noise(pitch), self.noise(yaw), round(self.noise(self.position.x),3), round(self.noise(self.position.y),3), round(self.noise(self.position.z),3),
            round(self.noise(self.yaw_error),3),round(self.noise(self.pitch_error),3), round(self.noise(x_error),3), round(self.noise(y_error),3), round(self.noise(z_error),3)]


        # article oceans, vecteur d'etat le plus rempli ; taille vecteur d'etat : 23
        state = state + [round(self.noise(self.linear_speed.x),3), round(self.noise(self.linear_speed.y),3), round(self.noise(self.linear_speed.z),3),
            round(self.noise(self.angular_speed.x),3), round(self.noise(self.angular_speed.y),3), round(self.noise(self.angular_speed.z),3),
            self.noise(roll), self.noise(pitch), self.noise(yaw), round(self.noise(self.position.x),3), round(self.noise(self.position.y),3), round(self.noise(self.position.z),3),
            round(self.noise(self.yaw_error),3),round(self.noise(self.pitch_error),3), round(self.noise(x_error),3), round(self.noise(y_error),3), round(self.noise(z_error),3)]


        '''
        # vecteur d'etat sans les positions en x,y,z ; taille vecteur d'etat : 20
        state = state + [round(self.noise(self.linear_speed.x),3), round(self.noise(self.linear_speed.y),3), round(self.noise(self.linear_speed.z),3),
            round(self.noise(self.angular_speed.x),3), round(self.noise(self.angular_speed.y),3), round(self.noise(self.angular_speed.z),3),
            self.noise(roll), self.noise(pitch), self.noise(yaw),
            round(self.noise(self.yaw_error),3),round(self.noise(self.pitch_error),3), round(self.noise(x_error),3), round(self.noise(y_error),3), round(self.noise(z_error),3)]
        '''

        '''
        # vecteur d'etat sans l'erreur en pitch et les angles d'euler ; taille vecteur d'etat : 16
        state = state + [round(self.noise(self.linear_speed.x),3), round(self.noise(self.linear_speed.y),3), round(self.noise(self.linear_speed.z),3),
            round(self.noise(self.angular_speed.x),3), round(self.noise(self.angular_speed.y),3), round(self.noise(self.angular_speed.z),3),
            round(self.noise(self.yaw_error),3), round(self.noise(x_error),3), round(self.noise(y_error),3), round(self.noise(z_error),3)]
        '''

        '''
        # vecteur d'etat sans les vitesses angulaires ; taille vecteur d'etat : 13
        state = state + [round(self.noise(self.linear_speed.x),3), round(self.noise(self.linear_speed.y),3), round(self.noise(self.linear_speed.z),3),
            round(self.noise(self.yaw_error),3), round(self.noise(x_error),3), round(self.noise(y_error),3), round(self.noise(z_error),3)]
        '''




        '''
        # current_distance au lieu de des erreurs en X, Y et Z ; taille vecteur d'etat : 11
        state = state + [round(self.noise(self.linear_speed.x),3), round(self.noise(self.linear_speed.y),3), round(self.noise(self.linear_speed.z),3),
            round(self.noise(self.yaw_error),3), round(self.noise(current_distance),3)]
        '''    

        # distance maximale entre deux sommets opposes de la boite definissant le world, avec x dans [-30,30], y dans [-30,30], z dans [-60,-1]
        current_distance_max = round(math.sqrt((-30.0 - 30.0)**2 + (-30.0 - 30.0)**2 + (-60.0 - -1.0)**2),2)

        # yaw_error appartient a [-pi,pi]

        '''
        # version normalisee, sauf pour les vitesses ; taille vecteur d'etat : 11
        state = state + [round(self.noise(self.linear_speed.x),3), round(self.noise(self.linear_speed.y),3), round(self.noise(self.linear_speed.z),3),
            round(self.noise(self.yaw_error/pi),3), round(self.noise(current_distance/current_distance_max),3)]
        '''

        # apres 1200 episodes : vitesse_max_x = 2.32 ; vitesse_max_y = 1.832 ; vitesse_max_z = 1.271

        vitesse_max_x = 2.32
        vitesse_max_y = 1.832
        vitesse_max_z = 1.271

        '''
        # version tout normalise ; taille vecteur d'etat : 11
        state = state + [round(self.noise(self.linear_speed.x/vitesse_max_x),3), round(self.noise(self.linear_speed.y/vitesse_max_y),3), round(self.noise(self.linear_speed.z/vitesse_max_z),3), round(self.noise(self.yaw_error/pi),3), round(self.noise(current_distance/current_distance_max),3)]
        '''





        '''
        # vecteur d'etat sans les vitesses lineaires ; taille vecteur d'etat : 10
        state = state + [round(self.noise(self.yaw_error),3), round(self.noise(x_error),3), round(self.noise(y_error),3), round(self.noise(z_error),3)]
        '''

        '''
        # vecteur d'etat sans l'erreur en cap (yaw) ; taille vecteur d'etat : 9
        state = state + [round(self.noise(x_error),3), round(self.noise(y_error),3), round(self.noise(z_error),3)]
        '''

        '''
        # vecteur d'etat sans les actions passees; taille vecteur d'etat : 4 
        state = [round(self.noise(self.yaw_error),3), round(self.noise(x_error),3), round(self.noise(y_error),3), round(self.noise(z_error),3)]
        '''


        '''
        # vecteur d'etat en remplacant les erreurs de posution par la pitch error ; taille vecteur d'etat : 8
        state = state + [round(self.noise(self.pitch_error),3), round(self.noise(self.yaw_error),3)]
        '''

        #print(state)

        '''
        # stock les 4 precedents etats et actions, plus les etats et actions actuels
	for a in state:
		self.obs.pop(0)
		self.obs.append(a)
        '''

        #return self.obs, done, reached, past_state
        return state_full, state, done, reached, past_state
        


    # Compute the reward according to state
    def setReward(self, state, done, reached, action, lambda1, lambda2, lambda3):

        # utiliser cette ligne pour le calcul de current_dist TOUT LE TEMPS, sauf si le vecteur d'etat est : current_distance au lieu de des erreurs en X, Y et Z ; taille vecteur d'etat : 11
        x_error = state[-3]
        y_error = state[-2]
        z_error = state[-1]
        current_distance = round(math.sqrt(x_error**2 + y_error**2 + z_error**2),2)
        
        # utiliser cette ligne pour le calcul de current_dist que si le vecteur d'etat est : current_distance au lieu de des erreurs en X, Y et Z ; taille vecteur d'etat : 11
        #current_distance = state[-1]


        reward_dist = 0
        nb_success = self.nb_success
        distance_rate = (self.past_distance - current_distance)

        #angular_speed = np.array( [ state[-14], state[-13], state[-12]] ) # valeur des indices pour le vecteur d'etat complet comme dans l'article oceans
        #angular_speed_norm = np.linalg.norm(angular_speed)

        # If the robot got closer to the target, a positive reward is send, otherwise a negative one is send
        if distance_rate > 0:
            #reward_dist = 25.0 * distance_rate
            #reward_dist = 200.0*distance_rate #- 10.0 * angular_speed_norm
            #reward_dist = 200.0*distance_rate - (10.0*(state[-5] + np.pi)) - 10.0 * angular_speed_norm
            #reward_dist = 40.0 * np.exp(-current_distance/73.0) # diviser par la distance maximale possible a la cible
            reward_dist = 40.0 * np.exp(-current_distance/20.0)
        if distance_rate <= 0:
            reward_dist = -10
        
        
        #reward_dist = lambda1 * np.exp(-(current_distance/100.0)**2)

        self.past_distance = current_distance

        self.actions_list.append(action)

        if len(self.actions_list) <= self.mean_actions_length:
            self.mean_actions = np.mean( np.asarray(self.actions_list) ,axis=0)

        else :
            self.mean_actions = np.add(self.mean_actions , (  np.subtract(action,self.actions_list[0]) )/self.mean_actions_length)
            self.actions_list.remove(self.actions_list[0])
        
        action_max = 240.0
        new_action_max = 100.0

        #reward_lambda2 = - lambda2 * np.sum(np.abs(action*new_action_max/action_max))
        #reward_lambda3 = - lambda3 * np.linalg.norm(np.subtract(self.mean_actions, action)*new_action_max/action_max)

        reward_lambda2 = - lambda2 * np.sum(np.abs(action))
        reward_lambda3 = - lambda3 * np.linalg.norm(np.subtract(self.mean_actions, action))

        reward_actuators = reward_lambda2 + reward_lambda3


        reward = reward_dist + reward_actuators


        if done: # Robot collide with something
            rospy.loginfo("Collision!!")
            reward = -550.
            self.pub_cmd_vel0.publish(FloatStamped())
            self.pub_cmd_vel1.publish(FloatStamped())
            self.pub_cmd_vel2.publish(FloatStamped())
            self.pub_cmd_vel3.publish(FloatStamped())
            self.pub_cmd_vel4.publish(FloatStamped())
            self.pub_cmd_vel5.publish(FloatStamped())

        if reached: # Robot have reach the target
            rospy.loginfo("Goal reached!!")
            reward = 500.
            nb_success = 1
            self.pub_cmd_vel0.publish(FloatStamped())
            self.pub_cmd_vel1.publish(FloatStamped())
            self.pub_cmd_vel2.publish(FloatStamped())
            self.pub_cmd_vel3.publish(FloatStamped())
            self.pub_cmd_vel4.publish(FloatStamped())
            self.pub_cmd_vel5.publish(FloatStamped())

        return reward, nb_success, reward_dist, reward_actuators, reward_lambda2, reward_lambda3
    
    def standStill(self):
        self.pub_cmd_vel0.publish(FloatStamped())
        self.pub_cmd_vel1.publish(FloatStamped())
        self.pub_cmd_vel2.publish(FloatStamped())
        self.pub_cmd_vel3.publish(FloatStamped())
        self.pub_cmd_vel4.publish(FloatStamped())
        self.pub_cmd_vel5.publish(FloatStamped())
		
    def step(self, action, past_action, past_state, ite, lambda1, lambda2, lambda3, ep, pid_buffer):

        self.cmd1.data = action[0]
        self.cmd2.data = action[1]
        self.cmd3.data = action[2]
        self.cmd4.data = action[3]
        self.cmd5.data = action[4]
        self.cmd6.data = action[5]

        if(not self.pid_mode) :
            self.pub_cmd_vel0.publish(self.cmd1)
            self.pub_cmd_vel1.publish(self.cmd2)
            self.pub_cmd_vel2.publish(self.cmd3)
            self.pub_cmd_vel3.publish(self.cmd4)
            self.pub_cmd_vel4.publish(self.cmd5)
            self.pub_cmd_vel5.publish(self.cmd6)

        data1 = None

        #rospy.sleep(0.1)


        #time1 = rospy.get_time()

        while data1 is None:
            try:
                #data1 = rospy.wait_for_message('/rexrov2/dvl_sonar0', Range, timeout=5)
                data1 = rospy.wait_for_message('/rexrov2/pressure', FluidPressure, timeout=1) # temps d'attente de 0.1 en simu temps reel
                #data1 = rospy.wait_for_message('/rexrov2/pose_gt', Odometry, timeout=1)  # temps d'attente de 0.05 en simu temps reel
            except:
                pass

        #time2 = rospy.get_time()
        #print time2 - time1
        
        state_full, state, done, reached, past_state = self.getState(data1, past_action, past_state, ite, ep)

        if(pid_buffer):
            action[0]=self.received_cmd0.data
            action[1]=self.received_cmd1.data
            action[2]=self.received_cmd2.data
            action[3]=self.received_cmd3.data
            action[4]=self.received_cmd4.data
            action[5]=self.received_cmd5.data

        #print(action)

        reward, nb_success, reward_dist, reward_actuators, reward_lambda2, reward_lambda3 = self.setReward(state, done, reached, action, lambda1, lambda2, lambda3)

        return np.asarray(state_full), np.asarray(state), reward, done, reached, nb_success, reward_dist, reward_actuators, reward_lambda2, reward_lambda3, past_state, self.goal_x, self.goal_y, self.goal_z, self.position.x, self.position.y, self.position.z, self.received_cmd0, self.received_cmd1, self.received_cmd2, self.received_cmd3, self.received_cmd4, self.received_cmd5

    def set_next_target(self):
        self.respawn_goal.next_goal(self.goal_x, self.goal_y)

    def reset(self, token, ite, ep):
        
        state_msg = ModelState()
        state_msg.model_name = 'rexrov2'
        state_msg.pose.position.x = 0.
        state_msg.pose.position.y = 0.
        state_msg.pose.position.z = -20.
        angle = randint(0, 360)
        q = quaternion_from_euler(0.0, 0.0, np.deg2rad(angle))
        state_msg.pose.orientation = Quaternion(*q)
        
        rospy.wait_for_service('/gazebo/reset_simulation')
        
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)

        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        
        # Get depths values from Depth-map
        data1 = None

        self.obs = [0.0] * 23 * 5

        #rospy.sleep(0.1)
        
        while data1 is None:
            try:
                #data1 = rospy.wait_for_message('/rexrov2/dvl_sonar0', Range, timeout=5)
                data1 = rospy.wait_for_message('/rexrov2/pressure', FluidPressure, timeout=1)
                #data1 = rospy.wait_for_message('/rexrov2/pose_gt', Odometry, timeout=1)
            except:
                pass
 
        if self.comparison_mode :

            # we reset and stop the PID controller node and we change the target, the SAC is working
            if self.j % 2 == 0 and (not self.full_pid_mode): 
                print("RL mode") 
                self.pid_mode = False
                
                rospy.wait_for_service('/rexrov2/set_pid_mode')
                try:
                    resp_set_pid_mode = self.set_pid_mode_proxy(False)
                    print("PID mode set to False")
                except (rospy.ServiceException) as e:
                    print("/rexrov2/set_pid_mode service call failed")

                '''
                rospy.wait_for_service('/rexrov2/reset_controller')
                try:
                    resp_reset_controller = self.reset_controller_proxy()
                    print("PID controller reset")
                except (rospy.ServiceException) as e:
                    print("/rexrov2/reset_controller service call failed")
                '''

                # Initialization of the models that haven't been load
                if self.initGoal:
                    i = True
                    self.goal_x, self.goal_y, self.goal_z = self.respawn_goal.getPosition(i)
                    self.initGoal = False
                else:
                    # Move the models, doesn't need to load them anymore
                    i = False
                    self.goal_x, self.goal_y, self.goal_z = self.respawn_goal.getPosition(i)
                

            # we start the PID controller, call the goto service and bypass the SAC
            else:
                print("PID mode")
                self.pid_mode = True
                
                rospy.wait_for_service('/rexrov2/set_pid_mode')
                try:
                    resp_set_pid_mode = self.set_pid_mode_proxy(True)
                    print("PID mode set to True")
                except (rospy.ServiceException) as e:
                    print("/rexrov2/set_pid_mode service call failed")

                if self.full_pid_mode:
                    # Initialization of the models that haven't been load
                    if self.initGoal:
                        i = True
                        self.goal_x, self.goal_y, self.goal_z = self.respawn_goal.getPosition(i)
                        self.initGoal = False
                    else:
                        # Move the models, doesn't need to load them anymore
                        i = False
                        self.goal_x, self.goal_y, self.goal_z = self.respawn_goal.getPosition(i)


                waypoint_msg = Waypoint()
                waypoint_msg.header.seq = 0
                waypoint_msg.header.stamp = rospy.Time.now()
                waypoint_msg.header.frame_id = 'world'
                waypoint_msg.point.x = self.goal_x
                waypoint_msg.point.y = self.goal_y
                waypoint_msg.point.z = self.goal_z
                waypoint_msg.max_forward_speed = 2.5
                waypoint_msg.heading_offset = 0.0
                waypoint_msg.use_fixed_heading = False
                waypoint_msg.radius_of_acceptance = 0.1

                rospy.wait_for_service('/rexrov2/go_to')
                try:
                    go_to_proxy = rospy.ServiceProxy('/rexrov2/go_to', GoTo)
                    go_to = go_to_proxy(waypoint_msg, 2.5,'lipb')
                except (rospy.ServiceException) as e:
                    print("/rexrov2/go_to call failed")
                
                '''
                os.system("rosservice call /rexrov2/go_to \"{waypoint: {header: {seq: 0, stamp: {secs: 0, nsecs: 0}, frame_id: 'world'}, point: {x: "+str(self.goal_x)+", y: "+str(self.goal_y)+", z: "+str(self.goal_z)+"}, max_forward_speed: 2.5, heading_offset: 0.0, use_fixed_heading: false, radius_of_acceptance: 0.1}, max_forward_speed: 2.5, interpolator: 'lipb'}\"")
                '''
             


                rospy.wait_for_service('/rexrov2/reset_controller')
                try:
                    resp_reset_controller = self.reset_controller_proxy()
                    print("PID controller reset")
                except (rospy.ServiceException) as e:
                    print("/rexrov2/reset_controller service call failed")
   
        else:
            
            # Initialization of the models that haven't been load
            if self.initGoal:
                self.goal_x, self.goal_y, self.goal_z = self.respawn_goal.getPosition(i=True)
                self.initGoal = False
            else:
                self.goal_x, self.goal_y, self.goal_z = self.respawn_goal.getPosition(i=False)



        # Initialize state
        past_state = [0] * 6

        self.goal_distance = self.getGoalDistace()
        state_full, state, done, reached, past_state = self.getState(data1, [0., 0., 0., 0., 0., 0.], past_state, ite, ep)

        self.mean_actions = [0.,0.,0.,0.,0.,0.]
        self.actions_list = []

        self.j += 1

        print("Target : ["+str(self.goal_x)+" ; "+str(self.goal_y)+" ; "+str(self.goal_z)+"]")

        return state_full, state, past_state, self.goal_x, self.goal_y, self.goal_z, self.position.x, self.position.y, self.position.z, self.pid_mode
