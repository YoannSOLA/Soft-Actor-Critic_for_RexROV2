

class PIDNN:

     # Class implementing a PID Neural Network (PIDNN)
     # Bibliography :
     # [1] Huailin Shu and Youguo Pi, PID neural networks for time-delay systems, Computers & Chemical Engineering, 2000
     # [2] Chengming Lee and Rongshun Chen, Optimal Self-Tuning PID Controller Based on Low Power Consumption for a Server Fan Cooling System, Sensors, 2015
     # [3] Zhen Zhang, Cheng Ma and Rong Zhu, Self-Tuning Fully-Connected PID Neural Network System for Distributed Temperature Sensing and Control of Instrument with Multi-Modules, Sensors 2016

     def __init__(self,kp=1.0,ki=0.0,kd=0.0,criterion=0,initial_y=0.0, learning_rate = 0.1, use_output=0,maximum_error=0.0):
           
           #initialization of the PIDNN parameters
           assert kp >= 0.0
           assert ki >= 0.0
           assert kd >= 0.0

           # In this special neural network there are no bias b1 and b2, only weight matrices W1 and W2.
           # Moreover, in order to reduce the computation time, the tracking error is directly given to the hidden layer, so no weight matrix W1 is necessary.
           # Finally, calculations are not vectorized in this implementation, because the input is scalar : the input layer is receiving data one by one. So no weight matrix W2 is necessary.
           self.kp=kp
           self.ki=ki
           self.kd=kd

           #initialization of the criterion used as the loss function by the neural network
           # 0 = current tracking error
           self.criterion=criterion

           # previous values of the outputs of the integral and derivative nodes
           self.hi_prev = 0.0
           self.hd_prev = 0.0

           # the list of all the tracking errors e_list
           self.e_list=[]

           # the number of iterations since the beginning
           self.iteration_number = 0.0

           # the list of all the values of the loss function
           self.loss_list=[]

           # the sum of all the squared-tracking errors (used during the computation of the loss function)
           self.e_squared_sum = 0.0

           # the sum of all the tracking errors (used during the backward propagation)
           self.e_sum = 0.0

           # previous values of the command u and the output y of the system
           # we use the initial value of the output y of the system
           self.y_prev = initial_y
           self.u_prev = 0.0

           # learning rate used by the gradient descent during the backpropagation
           self.learning_rate = learning_rate

           # a parameter that allows to use the output PIDNN as the input u of the system
           self.use_output = use_output

           # a parameter that tells if the pidnn needs to be turned off (no updates of the pid gains)
           self.turn_off = False

           # a parameter that tells the maximum error accepted : if the actual error is below this value, the pidnn is turned off
           self.maximum_error = maximum_error

     def get_pid_gains(self):
           return self.kp,self.ki,self.kd

     def get_e_list(self):
           return self.e_list

     def get_loss_list(self):
           return self.loss_list
           
     # We can change the criterion choosen for the computation of the loss function at any moment of the runtime
     def set_criterion(self,criterion):
           self.criterion=criterion

     def set_learning_rate(self,learning_rate):
           self.learning_rate=learning_rate

     def set_use_output(self,use_output):
           self.use_output=use_output

     def get_turn_off(self):
           return self.turn_off

     def set_turn_off(self,turn_off):
           self.turn_off=turn_off

     def get_maximum_error(self):
           return self.maximum_error

     def set_maximum_error(self,maximum_error):
           self.maximum_error=maximum_error

     # Sign function used during the backpropagation
     def sign(self,x):
           
           if x>0.0:
              result = 1.0
           elif x<0.0:
              result = -1.0
           else:
              result = 0.0

           return result

     # The forward propagation function of the PIDNN
     # Inputs : the tracking error e and the time step dt used to compute the derivative and the integral nodes
     # Outputs : the output of the PIDNN
     def forward_prop(self,e,dt):

           # We compute the outputs of the hidden layer

           # proportional node
           hp = e
           if e > 1.0 :
              hp = 1.0
           if e < -1.0 :
              hp = -1.0

           # integral node
           hi = (e + self.hi_prev)*dt
           if e > 1.0 :
              hi = 1.0
           if e < -1.0 :
              hi = -1.0
           self.hi_prev = hi

           # derivative node
           hd = (e - self.hd_prev)/dt
           if e > 1.0 :
              hd = 1.0
           if e < -1.0 :
              hd = -1.0
           self.hd_prev=hd

           output = self.kp * hp + self.ki * hi + self.kd * hd

           return output, hp, hi, hd

     # This function computes the loss function that has to be minimized by the PIDNN
     # Input : the current tracking error e
     # Output : the current value of the loss function
     def loss_function(self,e):
           
           # We compute the loss function based on the criterion choosen by the user

           # We add the square of the current error to the sum of all the previous squared-errors
           self.e_squared_sum = self.e_squared_sum + e**2
           #self.e_list.append(e)

           self.iteration_number = self.iteration_number + 1.0

           # Like in [1] : mean of all the squared-tracking errors since the beginning of the running
           if self.criterion == 1:
              loss = (1.0/self.iteration_number) * self.e_squared_sum
           # Like in [2] : sum of all the squared-tracking errors since the beginning of the running, multiply by 1/2
           elif self.criterion == 2:
              loss = 0.5 * self.e_squared_sum
           # Like in [3] : just the square of the current tracking error, multiply by 1/2
           else:
              loss = 0.5 * e**2

           #self.loss_list.append(loss)

           return loss

     # The backward propagation AND gradient descent function of the PIDNN
     # Inputs : the current tracking error e, the current input u and output y of the system, the outputs hp, hi and hd of the hidden layer of the PIDNN
     # Outputs : the new gains kp, ki and kd
     def backward_prop(self,e,u,y,hp,hi,hd):

           # We are going to compute all the partial derivative needed by the gradient descent.
           # For example, for Kp, the gradient descent is :
           #       Kp = Kp - alpha * dJ/dKp
           # According to the chain rule, we have :
           #       dJ/dKp = dJ/dY * dY/dU * dU/dKp

           self.e_sum = self.e_sum + e

           # Computation of dJ/dY
           if self.criterion == 1:
              dJ_dY = (-2.0/self.iteration_number) * self.e_sum
           elif self.criterion == 2:
              dJ_dY = - self.e_sum
           else:
              dJ_dY = -e

           # Computation of dY/dU
           dY_dU = self.sign( (y - self.y_prev) * (u - self.u_prev) )
           
           self.u_prev = u
           self.y_prev = y

           # Computation of dU/dKp, dU/dKi and dU/dKd
           dU_dKp = hp
           dU_dKi = hi
           dU_dKd = hd

           # Computation of dJ/dKp, dJ/dKi and dJ/dKd
           dJ_dKp = dJ_dY * dY_dU * dU_dKp
           dJ_dKi = dJ_dY * dY_dU * dU_dKi
           dJ_dKd = dJ_dY * dY_dU * dU_dKd

           # Application of the gradient descent to kp, ki and kd
           self.kp = self.kp - self.learning_rate * dJ_dKp
           self.ki = self.ki - self.learning_rate * dJ_dKi
           self.kd = self.kd - self.learning_rate * dJ_dKd

           return self.kp, self.ki, self.kd

     # Training function of the neural network : forward propagation, computation, backward propagation and gradient descent
     # Inputs : the current tracking error e, the time step dt (used to compute the derivative and the integral nodes), the current input u and output y of the system
     # Output : the output PIDNN, the current value of the loss function, the new gains kp, ki and kd
     def train(self,e,dt,y,u=0.0):

           output, hp, hi, hd = self.forward_prop(e,dt)
           loss = self.loss_function(e)

           if self.turn_off:
              kp, ki, kd = self.get_pid_gains()
           else:

              if abs(e) > self.maximum_error:
                 if self.use_output == 1:
                    kp, ki, kd = self.backward_prop(e,output,y,hp,hi,hd)
                 else:
                    kp, ki, kd = self.backward_prop(e,u,y,hp,hi,hd)
              else:
                 kp, ki, kd = self.get_pid_gains()

           return output, loss, kp, ki, kd

if __name__ == '__main__':

     print('Test of the PIDNN ')

     kp=2.0
     ki=3.0
     kd=4.0
     criterion=1
     initial_y=0.0
     learning_rate=0.1
     use_output=0

     pidnn=PIDNN(kp,ki,kd,criterion,initial_y,learning_rate,use_output)
     
     # a test without the train function

     r=0.75
     y=0.5
     e=r-y

     dt=1.0

     output, hp, hi, hd = pidnn.forward_prop(e,dt)
     print("output, hp, hi, hd = pidnn.forward_prop(e,dt) : ",output, hp, hi, hd)

     loss = pidnn.loss_function(e)
     print("loss = pidnn.loss_function(e) : ", loss)

     kp, ki, kd = pidnn.backward_prop(e,output,y,hp,hi,hd)
     print("kp, ki, kd = pidnn.backward_prop(e,output,y,hp,hi,hd)", kp, ki, kd)

     # a test with the train function
     
     pidnn.set_use_output(1)
     
     y=0.6
     e=r-y

     output, loss, kp, ki, kd = pidnn.train(e,dt,y)
     print("output, loss, kp, ki, kd = pidnn.train(e,dt,y,u)", output, loss, kp, ki, kd)

     y=0.7
     e=r-y

     output, loss, kp, ki, kd = pidnn.train(e,dt,y)
     print("output, loss, kp, ki, kd = pidnn.train(e,dt,y,u)", output, loss, kp, ki, kd)

     print("pidnn.get_loss_list()",pidnn.get_loss_list())
     print("pidnn.get_e_list()",pidnn.get_e_list())

     output, loss, kp, ki, kd = pidnn.train(0.0,dt,0.8)
     print("output, loss, kp, ki, kd = pidnn.train(e,dt,y,u)", output, loss, kp, ki, kd)

     y=0.9
     e=r-y

     pidnn.set_maximum_error(25.0)
     output, loss, kp, ki, kd = pidnn.train(e,dt,y)
     print("output, loss, kp, ki, kd = pidnn.train(e,dt,y,u)", output, loss, kp, ki, kd)

     y=0.8
     e=r-y

     pidnn.set_maximum_error(0.0)
     output, loss, kp, ki, kd = pidnn.train(e,dt,y)
     print("output, loss, kp, ki, kd = pidnn.train(e,dt,y,u)", output, loss, kp, ki, kd)

     y=0.7
     e=r-y

     pidnn.set_turn_off(True)
     output, loss, kp, ki, kd = pidnn.train(e,dt,y)
     print("output, loss, kp, ki, kd = pidnn.train(e,dt,y,u)", output, loss, kp, ki, kd)

     y=2
     e=r-y

     pidnn.set_turn_off(False)
     output, loss, kp, ki, kd = pidnn.train(e,dt,y)
     print("output, loss, kp, ki, kd = pidnn.train(e,dt,y,u)", output, loss, kp, ki, kd)
     
