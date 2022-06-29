import rospy
import os

from gazebo_msgs.srv import SpawnModel, SetModelState, GetModelState, SetModelStateRequest, DeleteModel
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Pose
from random import randint

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        
        self.username = 'solayo'

        # Target
        self.modelPath = '/home/'+self.username+'/.gazebo/models/cricket_ball/model.sdf'
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()

        self.j = 0

        self.goal_position = Pose()
        self.init_goal_x = 30.0
        self.init_goal_y = 30.0
        self.init_goal_z = -25.0

        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.goal_position.position.z = self.init_goal_z

        self.modelName  = 'goal'

        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self, i):
        while True:
            if not self.check_model:
                rospy.wait_for_service('/gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
                del_model_prox = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

                get_model_state_prox = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

                if not i:
                    del_model_prox('goal')
                    spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")

                if i:
                    spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")

                self.j += 1

                rospy.loginfo("Goal position : %.1f, %.1f, %.1f", self.goal_position.position.x, self.goal_position.position.y, self.goal_position.position.z)
                break
            else:
                pass

    def next_goal(self, last_x, last_y):
        r = randrange(1)
        if r == 0:
            self.goal_position.position.x = last_x - randint(5, 10)
        else:
            self.goal_position.position.x = last_x + randint(5, 10)

        r = randrange(1)
        if r == 0:
            self.goal_position.position.y = last_y - randint(5, 10)
        else:
            self.goal_position.position.y = last_x + randint(5, 10)

    def getPosition(self, i):


        self.goal_position.position.x = randint(-20, 20)

        if self.j % 2 == 0:
            self.goal_position.position.y = randint(-20, -5)
        else:
            self.goal_position.position.y = randint(5, 20)

        '''
        self.goal_position.position.x = randint(-50, 50)

        if self.j % 2 == 0:
            self.goal_position.position.y = randint(-50, -5)
        else:
            self.goal_position.position.y = randint(5, 50)
        '''


        self.goal_position.position.z = randint(-40, -10)

        # Update models position with new coordinates
        self.respawnModel(i)

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y
        self.last_goal_z = self.goal_position.position.z

        return self.goal_position.position.x, self.goal_position.position.y, self.goal_position.position.z
