import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim)

        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, inputs):
        #x = F.leaky_relu(self.linear1(inputs))
        #x = F.leaky_relu(self.linear2(x))

        x = F.leaky_relu(self.bn1(self.linear1(inputs)))
        x = F.leaky_relu(self.bn2(self.linear2(x)))

        x = self.linear3(x)

        return x

# Q value network avec les actions et les states qui sont concatene puis passent dans un dense

class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size)

        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        #x = F.leaky_relu(self.linear1(x))
        #x = F.leaky_relu(self.linear2(x))

        x = F.leaky_relu(self.bn1(self.linear1(x)))
        x = F.leaky_relu(self.bn2(self.linear2(x)))

        x = self.linear3(x)

        return x

'''
# Q value network avec les actions et les states qui passent d'abord dans deux denses independants avant d'etre concatenes
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.state = nn.Linear(num_inputs, 128)
        self.state.weight.data.uniform_(-init_w, init_w)
        self.state.bias.data.uniform_(-init_w, init_w)

        self.actions = nn.Linear(num_actions, 128)
        self.actions.weight.data.uniform_(-init_w, init_w)
        self.actions.bias.data.uniform_(-init_w, init_w)

        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):

        state = F.leaky_relu(self.state(state))
        action = F.leaky_relu(self.actions(action))
	
	#assert state.size(0) == action.size(0)	
	
        x = torch.cat([state, action], 1)
        x =  F.leaky_relu(self.linear1(x))
        x =  F.leaky_relu(self.linear2(x))
        x = self.linear3(x)

        return x
'''

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):

        #x = F.leaky_relu(self.linear1(state))
        #x = F.leaky_relu(self.linear2(x))

        x = F.leaky_relu(self.bn1(self.linear1(state)))
        x = F.leaky_relu(self.bn2(self.linear2(x)))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z).to(device)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample().to(device)
        action = torch.tanh(z).to(device)

        action = action.detach().cpu().numpy()

        return action[0]
