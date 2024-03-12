import torch
import torch.nn as nn
import torch.optim as optim


ALPHA = 1e-3    # learning rate 
TAU = 1e-3      # soft update parameter
GAMMA = 0.995   # discount factor 


'''
The class for the Q Neural Network.
'''
class QNetwork(nn.Module):

    # initialize the model
    def __init__(self, state_size, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_actions)

    # initialize the optimizer
    def init_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)

    # forward pass for prediction
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    # saves the model parameters
    def save(self, file_name):
        torch.save(self.state_dict(), file_name)


'''
Compute the loss, i.e. the MSE between the experienced rewards and the Q network predictions.
'''
def compute_loss(q_net, t_net, experiences):
    states, actions, rewards, next_states, done_vals = experiences
    
    # compute max Q^(s,a)
    max_qsa = torch.max((t_net(next_states)), dim=-1)[0]
    
    # set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (1 - done_vals) * GAMMA * max_qsa
    
    # get the q_values and reshape to match y_targets
    q_values = q_net(states)
    q_values = q_values[range(q_values.size(0)), actions.long()] 
        
    # compute the loss
    loss = nn.MSELoss()(y_targets, q_values)
    
    return loss


'''
Update the two networks based on experience.
'''
def updateModels(q_net, t_net, experiences):
    # calculate the loss
    loss = compute_loss(q_net, t_net, experiences)

    # zero the gradients
    q_net.optimizer.zero_grad()

    # backward pass
    loss.backward()

    # update the weights of q_network
    q_net.optimizer.step()

    # update the weights of target_q_network
    for target_param, param in zip(t_net.parameters(), q_net.parameters()):
        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
