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
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    def sample(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
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
    # 初始化缓冲区
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = []
 
    # 将一条经验数据添加到缓冲区中
    def push(self,state,action,reward,next_state):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state,action,reward,next_state))
 
    # 随机从缓冲区抽取batch_size大小的经验数据
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
                                         action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
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
        self.target_entropy = target_entropy  # 目标熵的大小
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

    def calc_target(self, rewards, next_states):  # 计算目标Q值
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

        # 更新两个Q网络
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

        # 更新策略网络
        new_actions, log_prob = self.actor.sample(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
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
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)


class GroundEnv:
    def __init__(self,size,rho):
        self.size = size
        self.rho = rho
        self.Nagent = int(size**2*rho)
        self.position,self.v = self.generate(self.size,self.Nagent)
        self.kBT = 0.0005
        self.tstep = 0.05
        self.max_neighbor = sense_neighbor
        self.states = np.zeros([self.Nagent,self.max_neighbor*4+1])
        self.reward = np.zeros(self.Nagent)
        self.Frep = np.zeros([2,self.Nagent])
        self.strength = np.ones(self.Nagent)

    def reset(self):
        self.position,self.v = self.generate(self.size,self.Nagent)
        self.strength = np.ones(self.Nagent)

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

    def step(self,omega,acceleration):
        # 执行动作
        ksi = np.random.normal(0,np.sqrt(2*self.kBT),[2,self.Nagent])
        direction = np.arctan2(self.v[1],self.v[0])
        Fsmt = np.zeros([2,self.Nagent])
        Fsmt[0] = np.cos(direction+omega*self.tstep)
        Fsmt[1] = np.sin(direction+omega*self.tstep)
        self.strength += acceleration*self.tstep
        self.strength = np.clip(self.strength,0,1)
        self.v = Fsmt*self.strength + ksi
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
    # 生成迷宫图像
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
#damp = 0.2
#strength = 0.2
action_bound = np.pi
#action_bound = strength
target_entropy = np.log(action_bound*2)
ground_size = 40
sense_radius = 4
input_shape = state_dim
#agt = DQNet()
#agt(torch.from_numpy(np.zeros(input_shape)).float())
num_actions = 1

actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
hidden_dim = 256
gamma = 0.99
agttau = 0.005  # 软更新参数·
rho = 0.1
agt = SACAgent(state_dim, hidden_dim, action_dim, action_bound,actor_lr, critic_lr, alpha_lr, target_entropy, agttau, gamma, device)
for count in np.arange(10):
    reward0 = 6.5
    agtpath = f'./1127/systemerror/count={count}'
    grd = GroundEnv(ground_size,rho)
    NN = grd.Nagent
    tt = 1000
    NT = int(tt/grd.tstep)
    tau = 20
    tt = int(tt/(grd.tstep*tau))
    for count0 in range(2):
        systemerror = count0*0.2+0.1
        agt.load(agtpath+f'/{round(systemerror,1)}',99)
        for noise in [0.0]:
            path = f'./1216/condensation/systemerror={round(systemerror,1)}'
            mkdir(path+f'/{count}')
            grd.kBT = 10**(noise-3)/2
            print(f'systemerror={round(systemerror,1)}/')
            action = (np.random.random(NN)-0.5)*action_bound*2
            positions = np.zeros([NT,2,NN])
            velocitys = np.zeros([NT,2,NN])
            acc = np.zeros(NN)
            grd.reset()
            for relax in range(20):
                grd.step(np.zeros(NN),np.zeros(NN))
            for it in range(NT):
                if it%tau==0:
                    if it%2000==0:
                        print('it=',it)
                    prandom = np.random.random(NN)
                    #prandom = np.ones(NN)
                    for agent in np.arange(NN):
                        if prandom[agent]>=systemerror:
                            action[agent] = agt.get_action(grd.states[agent])
                        else:
                            action[agent] = (np.random.random()-0.5)*np.pi*2
                positions[it] = grd.position
                velocitys[it] = grd.v
                acc = 0.5-grd.reward/reward0
                grd.step(action,acc)
                if it%1000==0:
                    np.save(path+f'/{count}/trace.npy',positions)
                    np.save(path+f'/{count}/v.npy',velocitys)
            np.save(path+f'/{count}/trace.npy',positions)
            np.save(path+f'/{count}/v.npy',velocitys)

