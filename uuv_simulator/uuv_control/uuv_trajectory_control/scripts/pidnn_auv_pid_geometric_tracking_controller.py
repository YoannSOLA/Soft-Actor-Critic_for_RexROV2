#!/usr/bin/env python

import rospy

import uuv_control_msgs.msg as uuv_control_msgs
import uuv_control_msgs.srv as uuv_control_srv

from std_msgs.msg import Float64, Bool
#import std_msgs.msg 

from uuv_control_msgs.msg import PIDNNInput
from uuv_control_msgs.srv import SetPIDParams, SetTurnOffPIDNN, SetTurnOffPIDNNResponse, GetTurnOffPIDNN, GetTurnOffPIDNNResponse, SetMaximumErrorPIDNN, SetMaximumErrorPIDNNResponse, GetMaximumErrorPIDNN, GetMaximumErrorPIDNNResponse
from uuv_control_interfaces import pidnn

class PIDNNAUVPIDGeometricTrackingController:

    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize PIDNN tuning for vehicle <%s>' % self.namespace)
        

        # Reading the initialization of the thruster PIDNN gains
        self.initial_p_thrust = rospy.get_param('~initial_p_thrust', 0.0)
        assert self.initial_p_thrust >= 0

        self.initial_i_thrust = rospy.get_param('~initial_i_thrust', 0.0)
        assert self.initial_i_thrust >= 0

        self.initial_d_thrust = rospy.get_param('~initial_d_thrust', 0.0)
        assert self.initial_d_thrust >= 0

        # Reading the initialization of the roll PIDNN gains
        self.initial_p_roll = rospy.get_param('~initial_p_roll', 0.0)
        assert self.initial_p_roll >= 0

        self.initial_i_roll = rospy.get_param('~initial_i_roll', 0.0)
        assert self.initial_i_roll >= 0

        self.initial_d_roll = rospy.get_param('~initial_d_roll', 0.0)
        assert self.initial_d_roll >= 0

        # Reading the initialization of the pitch PIDNN gains
        self.initial_p_pitch = rospy.get_param('~initial_p_pitch', 0.0)
        assert self.initial_p_pitch >= 0

        self.initial_i_pitch = rospy.get_param('~initial_i_pitch', 0.0)
        assert self.initial_i_pitch >= 0

        self.initial_d_pitch = rospy.get_param('~initial_d_pitch', 0.0)
        assert self.initial_d_pitch >= 0

        # Reading the initialization of the yaw PIDNN gains
        self.initial_p_yaw = rospy.get_param('~initial_p_yaw', 0.0)
        assert self.initial_p_yaw >= 0

        self.initial_i_yaw = rospy.get_param('~initial_i_yaw', 0.0)
        assert self.initial_i_yaw >= 0

        self.initial_d_yaw = rospy.get_param('~initial_d_yaw', 0.0)
        assert self.initial_d_yaw >= 0


        # Reading the initial state of the AUV
        self.initial_thrust = rospy.get_param('~initial_thrust', 0.0)

        self.initial_roll = rospy.get_param('~initial_roll', 0.0)

        self.initial_pitch = rospy.get_param('~initial_pitch', 0.0)

        self.initial_yaw = rospy.get_param('~initial_yaw', 0.0)
    
 
        # Reading parameters used during the training of PIDNNs
        self.criterion = rospy.get_param('~criterion', 0)

        self.learning_rate = rospy.get_param('~learning_rate', 0.1)
        assert self.learning_rate >= 0

        
        # Initialization of 3 PIDNN : thrust, pitch and yaw.
        # At the moment, the roll doesn't need to be fine-tuned in the control of the AUV (only a P controller roughly tuned in order to keep the roll equal to 0 degree)
        self.pidnn_thrust=pidnn.PIDNN(self.initial_p_thrust, self.initial_i_thrust, self.initial_d_thrust, self.criterion, self.initial_thrust, self.learning_rate, use_output = 0)
        
        self.pidnn_pitch=pidnn.PIDNN(self.initial_p_pitch, self.initial_i_pitch, self.initial_d_pitch, self.criterion, self.initial_pitch, self.learning_rate, use_output = 0)

        self.pidnn_yaw=pidnn.PIDNN(self.initial_p_yaw, self.initial_i_yaw, self.initial_d_yaw, self.criterion, self.initial_yaw, self.learning_rate, use_output = 0)


        # Time step
        self.dt = 0
        self.prev_time = rospy.get_time()


        # A parameter allowing to use the real dt or dt = 1, during the training of the PIDNNs
        self.use_dt = rospy.get_param('~use_dt', 0)


        # Subscriber to the PIDNNInput topic (given by the PID controller of the AUV)
        self.pidnn_input_sub = rospy.Subscriber('pidnn_input', PIDNNInput, self.pidnn_input_callback)

        self.set_thrust_turn_off_service = rospy.Service('set_thrust_turn_off',SetTurnOffPIDNN,self.set_thrust_turn_off_callback)
        self.get_thrust_turn_off_service = rospy.Service('get_thrust_turn_off',GetTurnOffPIDNN,self.get_thrust_turn_off_callback)
        self.set_thrust_maximum_error_service = rospy.Service('set_thrust_maximum_error',SetMaximumErrorPIDNN,self.set_thrust_maximum_error_callback)
        self.get_thrust_maximum_error_service = rospy.Service('get_thrust_maximum_error',GetMaximumErrorPIDNN,self.get_thrust_maximum_error_callback)

        self.set_pitch_turn_off_service = rospy.Service('set_pitch_turn_off',SetTurnOffPIDNN,self.set_pitch_turn_off_callback)
        self.get_pitch_turn_off_service = rospy.Service('get_pitch_turn_off',GetTurnOffPIDNN,self.get_pitch_turn_off_callback)
        self.set_pitch_maximum_error_service = rospy.Service('set_pitch_maximum_error',SetMaximumErrorPIDNN,self.set_pitch_maximum_error_callback)
        self.get_pitch_maximum_error_service = rospy.Service('get_pitch_maximum_error',GetMaximumErrorPIDNN,self.get_pitch_maximum_error_callback)

        self.set_yaw_turn_off_service = rospy.Service('set_yaw_turn_off',SetTurnOffPIDNN,self.set_yaw_turn_off_callback)
        self.get_yaw_turn_off_service = rospy.Service('get_yaw_turn_off',GetTurnOffPIDNN,self.get_yaw_turn_off_callback)
        self.set_yaw_maximum_error_service = rospy.Service('set_yaw_maximum_error',SetMaximumErrorPIDNN,self.set_yaw_maximum_error_callback)
        self.get_yaw_maximum_error_service = rospy.Service('get_yaw_maximum_error',GetMaximumErrorPIDNN,self.get_yaw_maximum_error_callback)


    def set_thrust_turn_off_callback(self, request):
        print("Setting the thrust PIDNN turn_off parameter to : "+str(request.turn_off)) 
        self.pidnn_thrust.set_turn_off(request.turn_off)
        return SetTurnOffPIDNNResponse(True)

    def get_thrust_turn_off_callback(self, request):
        return GetTurnOffPIDNNResponse(self.pidnn_thrust.get_turn_off())

    def set_thrust_maximum_error_callback(self, request):
        print("Setting the thrust PIDNN maximum_error parameter to : "+str(request.maximum_error)) 
        self.pidnn_thrust.set_maximum_error(request.maximum_error)
        return SetMaximumErrorPIDNNResponse(True)

    def get_thrust_maximum_error_callback(self, request):
        return GetMaximumErrorPIDNNResponse(self.pidnn_thrust.get_maximum_error())


    def set_pitch_turn_off_callback(self, request):
        print("Setting the pitch PIDNN turn_off parameter to : "+str(request.turn_off)) 
        self.pidnn_pitch.set_turn_off(request.turn_off)
        return SetTurnOffPIDNNResponse(True)

    def get_pitch_turn_off_callback(self, request):
        return GetTurnOffPIDNNResponse(self.pidnn_pitch.get_turn_off())

    def set_pitch_maximum_error_callback(self, request):
        print("Setting the pitch PIDNN maximum_error parameter to : "+str(request.maximum_error)) 
        self.pidnn_pitch.set_maximum_error(request.maximum_error)
        return SetMaximumErrorPIDNNResponse(True)

    def get_pitch_maximum_error_callback(self, request):
        return GetMaximumErrorPIDNNResponse(self.pidnn_pitch.get_maximum_error())


    def set_yaw_turn_off_callback(self, request):
        print("Setting the yaw PIDNN turn_off parameter to : "+str(request.turn_off)) 
        self.pidnn_yaw.set_turn_off(request.turn_off)
        return SetTurnOffPIDNNResponse(True)

    def get_yaw_turn_off_callback(self, request):
        return GetTurnOffPIDNNResponse(self.pidnn_yaw.get_turn_off())

    def set_yaw_maximum_error_callback(self, request):
        print("Setting the yaw PIDNN maximum_error parameter to : "+str(request.maximum_error)) 
        self.pidnn_yaw.set_maximum_error(request.maximum_error)
        return SetMaximumErrorPIDNNResponse(True)

    def get_yaw_maximum_error_callback(self, request):
        return GetMaximumErrorPIDNNResponse(self.pidnn_yaw.get_maximum_error())


    def update_time_step(self):
        t = rospy.get_time()
        self.dt = t - self.prev_time
        self.prev_time = t


    def pidnn_input_callback(self, pidnn_input):

        #t_beginning = rospy.get_time()

        # Updating the time step
        self.update_time_step()

        #print("[PIDNN] dt = " + str(self.dt))

        e = pidnn_input.e
        u = pidnn_input.u
        y = pidnn_input.y

        #print("PIDNN : Receiving inputs",e,u,y)

        if len(e) == 4 and len(u) == 4 and len(y) == 4:

            e_thrust = e[0]
            e_roll = e[1]
            e_pitch = e[2]
            e_yaw = e[3]

            u_thrust = u[0]
            u_roll = u[1]
            u_pitch = u[2]
            u_yaw = u[3]

            y_thrust = y[0]
            y_roll = y[1]
            y_pitch = y[2]
            y_yaw = y[3]

            if self.use_dt == 1:
                training_dt = self.dt
            else:
                training_dt = 1

            output_thrust, loss_thrust, kp_thrust, ki_thrust, kd_thrust = self.pidnn_thrust.train(e_thrust,training_dt,y_thrust,u_thrust)

            output_pitch, loss_pitch, kp_pitch, ki_pitch, kd_pitch = self.pidnn_pitch.train(e_pitch,training_dt,y_pitch,u_pitch)

            output_yaw, loss_yaw, kp_yaw, ki_yaw, kd_yaw = self.pidnn_yaw.train(e_yaw,training_dt,y_yaw,u_yaw)

            Kp = [kp_thrust, self.initial_p_roll, kp_pitch, kp_yaw]
            Kd = [kd_thrust, self.initial_d_roll, kd_pitch, kd_yaw]
            Ki = [ki_thrust, self.initial_i_roll, ki_pitch, ki_yaw]

            # Calling of the service setting the PID gains of the controller of the AUV
            success = self.set_pid_params_client(Kp,Kd,Ki)

            #print("----------------- PIDNN : Success of the service calling",success)

            #t_end = rospy.get_time()
            #print("[PIDNN] One loop = " + str(t_end - t_beginning))

    def set_pid_params_client(self,Kp,Kd,Ki):

        #print("PIDNN : Setting new PID gains",Kp,Kd,Ki)

        rospy.wait_for_service('set_pid_params')
        try:
            set_pid_params = rospy.ServiceProxy('set_pid_params', SetPIDParams)
            response = set_pid_params(Kp,Kd,Ki)
            return response.success
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

if __name__ == '__main__':
    print('Starting AUV PIDNN trajectory tracker')
    rospy.init_node('pidnn_auv_pid_geometric_tracking_controller')

    try:
        node = PIDNNAUVPIDGeometricTrackingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
