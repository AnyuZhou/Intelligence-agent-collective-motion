import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions import Normal
from scipy.spatial import Voronoi
import os
#import matplotlib.pylab as plt
# Deep Q Network


def mkdir(path):
    # 引入模块
 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
 
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x)) + 1e-6
        return mu, std
    
    def sample(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
	log_prob = log_prob - torch.log(self.action_bound.abs() + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class ReplayBuffer:

    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = []
 

    def push(self,state,action,reward,next_state):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state,action,reward,next_state))
 

    def sample(self,batch_size):
        states,actions,rewards,next_states = zip(*random.sample(self.buffer,batch_size))
        return states,actions,rewards,next_states
 
    def __len__(self):
        return len(self.buffer)
class SACAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device) 
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(1e-2), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.memory = ReplayBuffer(capacity=100000)
        self.batch_size = 4096
        
    def remember(self,states,actions,rewards,next_states):
        for i in range(len(actions)):
            self.memory.push(states[i],actions[i],rewards[i],next_states[i])

    def get_action(self, state):
        state = torch.from_numpy(np.array(state)).float()
        action = self.actor.sample(state)[0]
        return action.item()

    def calc_target(self, rewards, next_states):  
        next_actions, log_prob = self.actor.sample(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        states,actions,rewards,next_states = self.memory.sample(self.batch_size)
        states = torch.from_numpy(np.array(states)).float()
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1)
        next_states = torch.from_numpy(np.array(next_states)).float()
        actions = torch.from_numpy(np.array(actions)).float().unsqueeze(1)

        td_target = self.calc_target(rewards, next_states)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_actions, log_prob = self.actor.sample(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
       

    def save(self,path):
        torch.save(self.critic_1.state_dict(),path+"q1_model.pt")
        torch.save(self.critic_2.state_dict(),path+"q2_model.pt")
        torch.save(self.actor.state_dict(),path+"policy_model.pt")
 
    def load(self,path,epoch):
        self.critic_1.load_state_dict(torch.load(path+f"/epoch={epoch}/q1_model.pt"))
        self.critic_2.load_state_dict(torch.load(path+f"/epoch={epoch}/q2_model.pt"))
        self.actor.load_state_dict(torch.load(path+f"/epoch={epoch}/policy_model.pt"))
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.log_alpha = torch.tensor(np.log(np.load(path+'/alpha.npy')[epoch]), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)


class GroundEnv:
    def __init__(self,size,rho):
        self.size = size
        self.rho = rho
        self.Nagent = int(size**2*rho)
        self.position,self.v = self.generate(self.size,self.Nagent)
        self.kBT = 0.005
        self.damp = 10
        self.tstep = 0.05
        self.max_neighbor = sense_neighbor
        self.states = np.zeros([self.Nagent,self.max_neighbor*4+1])
        self.reward = np.zeros(self.Nagent)
        self.Frep = np.zeros([2,self.Nagent])

    def reset(self):
        self.position,self.v = self.generate(self.size,self.Nagent)

    def repulsion(self,indexi,indexj):
        lam = 50
        B = lam/(lam-1)
        A = lam*B**(lam-1)
        dx,dy = (self.position[:,indexj] - self.position[:,indexi] + self.size/2)%self.size - self.size/2
        dr = np.sqrt(dx**2+dy**2)
        if dr>=B:
            F = 0
        else:
            F = A*((lam-1)/dr**lam-lam/dr**(lam+1))
        F = np.clip(-2,F,0)
        phi = np.arctan2(dy,dx)
        return F*np.array([np.cos(phi),np.sin(phi)])

    def step(self,actions):
        ksi = np.random.normal(0,np.sqrt(2*self.kBT*self.damp),[2,self.Nagent])
        direction = np.arctan2(self.v[1],self.v[0])
        Fsmt = np.zeros([2,self.Nagent])
        Fsmt[0] = np.cos(direction+actions*self.tstep)*self.damp
        Fsmt[1] = np.sin(direction+actions*self.tstep)*self.damp
        self.v = 1/self.damp*(Fsmt + ksi)
        self.position += (self.v+self.Frep) * self.tstep
        self.position = self.position%self.size
        self.Frep = self.observe()
        return self.position
    
    def observe(self):
        env = [[[] for i in range(self.size)] for j in range(self.size)]
        F_rep = np.zeros([2,self.Nagent])
        posx = np.zeros(self.Nagent,dtype=int)
        posy = np.zeros(self.Nagent,dtype=int)
        direction = np.arctan2(self.v[1],self.v[0])
        velocity = np.linalg.norm(self.v,axis=0)
        for agent in range(self.Nagent):
            posx[agent] = int(self.position[0][agent]%self.size)
            posy[agent] = int(self.position[1][agent]%self.size)
            env[posx[agent]][posy[agent]].append(agent)
        for agent in range(self.Nagent):
            dr = []
            dtheta = []
            dphi = []
            dv = []
            voronoi_position = [[0,0]]
            for i in range(2*sense_radius+1):
                for j in range(2*sense_radius+1):
                    for neighbor in env[(posx[agent]+i-sense_radius)%self.size][(posy[agent]+j-sense_radius)%self.size]:
                        if neighbor!=agent:
                            dx,dy = (self.position[:,neighbor] - self.position[:,agent] + self.size/2)%self.size - self.size/2
                            dr.append(np.sqrt(dx**2+dy**2))
                            voronoi_position.append([dx,dy])
                            dtheta.append(np.arctan2(dy,dx))
                            dphi.append(direction[neighbor])
                            dv.append(velocity[neighbor])
                            frep = self.repulsion(agent,neighbor)
                            F_rep[:,agent] += frep
            dr = np.array(dr)
            dv = np.array(dv)
            dtheta = np.array(dtheta)-direction[agent]
            dphi = np.array(dphi)-direction[agent]
            self.reward[agent] = np.sum(np.exp(-dr))
            #self.reward[agent] = np.sum(dr**-6)
            if len(dr)>=3:
                vor = Voronoi(voronoi_position)
                ridge = vor.ridge_points
                index = []
                for edge in ridge:
                    if edge[0]==0:
                        index.append(edge[1]-1)
                    if edge[1]==0:
                        index.append(edge[0]-1)
                dr = dr[index]
                dv = dv[index]
                dtheta = dtheta[index]
                dphi = dphi[index]
            sort = dr.argsort()
            if len(dr)<=self.max_neighbor:
                self.states[agent,:]=0
                self.states[agent,1:len(dr)*4+1:4] = (dr*np.cos(dtheta))[sort]
                self.states[agent,2:len(dr)*4+1:4] = (dr*np.sin(dtheta))[sort]
                self.states[agent,3:len(dr)*4+1:4] = (dv*np.cos(dphi))[sort]
                self.states[agent,4:len(dr)*4+1:4] = (dv*np.sin(dphi))[sort]
            else:

                self.states[agent,1::4] = (dr*np.cos(dtheta))[sort][:self.max_neighbor]
                self.states[agent,2::4] = (dr*np.sin(dtheta))[sort][:self.max_neighbor]
                self.states[agent,3::4] = (dv*np.cos(dphi))[sort][:self.max_neighbor]
                self.states[agent,4::4] = (dv*np.sin(dphi))[sort][:self.max_neighbor]
        self.states[:,0] = velocity
        return F_rep

    @staticmethod
    def generate(size,Nagent):
        position = np.zeros([2,Nagent])
        v = np.zeros([2,Nagent])
        for i in range(Nagent):
            position[:,i] = np.random.random(2)*size
            direction = np.random.random()*np.pi*2
            v[0,i] = np.cos(direction)
            v[1,i] = np.sin(direction)
            #direction[i] = -np.pi
        return position,v

sense_neighbor = 16
state_dim = sense_neighbor*4+1
action_dim = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

action_bound = np.pi

target_entropy = -action_dim
ground_size = 40
sense_radius = 4
input_shape = state_dim

num_actions = 1

actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
hidden_dim = 256
gamma = 0.99
agttau = 0.005  
rho = 0.1
for count in range(10):
    intelligence = 1.0
    systemerror = 1-intelligence
    reward0 = 6.5
    agt = SACAgent(state_dim, hidden_dim, action_dim, action_bound,actor_lr, critic_lr, alpha_lr, target_entropy, agttau, gamma, device)
    grd = GroundEnv(ground_size,rho)
    num_epoch = 100
    NN = grd.Nagent
    tt = 200
    NT = int(tt/grd.tstep)
    tau = 20
    tt = int(tt/(grd.tstep*tau))
    action = (np.random.random(NN)-0.5)*action_bound*2
    path = f'filename'
    alpha = np.zeros(num_epoch)
    for epoch in np.arange(num_epoch):
        mkdir(path+f'/epoch={epoch}')
        grd.reset()
        #Fspp = np.zeros([2,NN])
        for relax in range(20):
            grd.step(np.zeros(NN))
        step = 0
        it = 0
        positions = np.zeros([NT,2,NN])
        velocitys = np.zeros([NT,2,NN])
        actions = np.zeros([tt,NN])
        rewards = np.zeros([tt,NN])
        grd.observe()
        state = grd.states
        if epoch>0:
            pagent = np.random.random(NN)
            for agent in range(NN):
                if pagent[agent]>systemerror:
                    action[agent] = agt.get_action(state[agent])
                else:
                    action[agent] = (np.random.random()-0.5)*action_bound*2
        next_state = None
        actions[it] = action
        while(step<NT):
            if step%tau==0:
                if it%20==0:
                    print('it=',it)
                if it!=0:
                    next_state = grd.states
                    reward = (grd.reward/reward0*(1-systemerror)+np.random.random(NN)*systemerror-0.2)*10
                    agt.remember(state,action,reward,next_state)
                    agt.train()
                    state = next_state.copy()
                    pagent = np.random.random(NN)
                    for agent in range(NN):
                        if pagent[agent]>systemerror:
                            action[agent] = agt.get_action(state[agent])
                        else:
                            action[agent] = (np.random.random()-0.5)*action_bound*2

                    actions[it] = action
                    rewards[it] = reward
                it += 1 
            positions[step] = grd.position
            velocitys[step] = grd.v
            grd.step(action)
            step += 1
        alpha[epoch] = agt.log_alpha.exp()
        agt.save(path+f'/epoch={epoch}/')
        #np.save(path+f'/epoch={epoch}/reward.npy',rewards)
        #np.save(path+f'/epoch={epoch}/trace.npy',positions)
        #np.save(path+f'/epoch={epoch}/v.npy',velocitys)
        #np.save(path+f'/epoch={epoch}/action.npy',actions)
        np.save(path+f'/alpha.npy',alpha)
