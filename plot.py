# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 10:50:11 2025

@author: Administrator
"""
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.distributions import Normal
from scipy.spatial import Voronoi
import os
import ffmpeg
import matplotlib.pylab as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import matplotlib
matplotlib.rcParams['font.family'] = 'Times new Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
colors2 = ['#d8c0c3','#4a5656','#9b394b','#99CDE7','#868d2a','#006600']
colors = ['#d8c0c3','#4a5656','#9b394b','#002CB0','#99CDE7','#868d2a','#006600']
colors0 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',   # 使用颜色编码定义颜色
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
klabel = [0.0,0.2,0.4,0.5,0.6,0.8,1.0]
rho = 0.1
"""
#snapshots in Fig.3, 4 and 7
def circle(x,y,r):
    theta = np.linspace(0,np.pi*2,100)
    dot = np.zeros([2,100])
    dot[0] = (x + np.cos(theta)*r)%ground_size
    dot[1] = (y + np.sin(theta)*r)%ground_size
    return dot
def update(frame):
    it = tlist[frame]
    for i in range(NN):
        dot[:,i*100:i*100+100] = circle(positions[it][0][i],positions[it][1][i],0.5)
    dots.set_offsets(np.c_[dot[0],dot[1]])
    text.set_text('time= {}'.format(round(it*0.05,2)))
    return dots,

import matplotlib.animation as animation

p=0
ground_size=40
NN = 160
tlist = np.arange(0,20000,40)
for ierror in np.arange(5,6):
    p=0
    error=1-klabel[ierror]
    positions = np.load(f'./systemerror={round(error,1)}/0/trace.npy')
    positions[:,0,:] += -10 #10
    positions[:,1,:] += -6 #3
    positions = positions%ground_size
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0.2,0.2,0.6,0.6])
    for spine in ax.spines.values():
        spine.set_linewidth(4)
    dot = np.zeros([2,NN*100])
    for i in range(NN):
        it = -1
        dot[:,i*100:i*100+100] = circle(positions[it][0][i],positions[it][1][i],0.5)
    dots = ax.scatter(dot[0],dot[1],color=colors[ierror],marker='o',s=0.5)
    text = ax.text(0.05, 1.02, r'$k = $'+f'{round(1-error,1)}',fontsize=54,transform=ax.transAxes)
    ax.xaxis.set_tick_params(labelsize=54)
    ax.yaxis.set_tick_params(labelsize=54)
    ax.set_xlabel('            X',fontsize=54)
    ax.set_ylabel('         Y',fontsize=54)
    ax.set_xlim(0,ground_size)
    ax.set_ylim(0,ground_size)
    ax.xaxis.set_ticks(np.linspace(0,ground_size,3))
    ax.yaxis.set_ticks(np.linspace(0,ground_size,3))
    ax.tick_params(axis='both',which='major',labelsize=54,width=4,length=8)
    plt.show()
    fig.savefig(f'./intell,k={round(error,1)}kBT={p-3}.png', dpi=400, bbox_inches='tight')


ground_size=40
NN = 160
tlist = np.arange(0,20000,40)
for it in [0,400,800]:
    #it=-1
    positions = np.load(f'./trace.npy')

    positions = positions%ground_size
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0.2,0.2,0.6,0.6])
    for spine in ax.spines.values():
        spine.set_linewidth(4)
    dot = np.zeros([2,NN*100])
    for i in range(NN):
        #it = -1
        dot[:,i*100:i*100+100] = circle(positions[it][0][i],positions[it][1][i],0.5)
    dots = ax.scatter(dot[0],dot[1],color=colors[6],marker='o',s=0.5)
    text = ax.text(0.05, 1.02, 'Time= {}'.format(round(it*0.05,2)),fontsize=54,transform=ax.transAxes)  
    ax.xaxis.set_tick_params(labelsize=54)
    ax.yaxis.set_tick_params(labelsize=54)
    ax.set_xlabel('            X',fontsize=54)
    ax.set_ylabel('         Y',fontsize=54)
    #ax.text(0.05*ground_size,ground_size*0.05,s=f'k={round(1-p*0.2,1)}',color='k',fontsize=18)
    ax.set_xlim(0,ground_size)
    ax.set_ylim(0,ground_size)
    ax.xaxis.set_ticks(np.linspace(0,ground_size,3))
    ax.yaxis.set_ticks(np.linspace(0,ground_size,3))
    ax.tick_params(axis='both',which='major',labelsize=54,width=4,length=8)
    plt.show()
    fig.savefig(f'./time={it}.png', dpi=400, bbox_inches='tight')


ground_size=40
NN = 160
tlist = np.arange(0,20000,40)
for epoch in [59,99]:
    positions = np.load(f'./systemerror/{0.0}/epoch={epoch}/trace.npy')
    positions[:,0,:] += 14   #20&100
    positions[:,1,:] += 5
    positions = positions%ground_size
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0.2,0.2,0.6,0.6])
    for spine in ax.spines.values():
        spine.set_linewidth(4)
    dot = np.zeros([2,NN*100])
    for i in range(NN):
        it = -1
        dot[:,i*100:i*100+100] = circle(positions[it][0][i],positions[it][1][i],0.5)
    dots = ax.scatter(dot[0],dot[1],color=colors[6],marker='o',s=0.5)
    text = ax.text(0.05, 1.02, f'Episode={epoch+1}',fontsize=54,transform=ax.transAxes)
    ax.xaxis.set_tick_params(labelsize=54)
    ax.yaxis.set_tick_params(labelsize=54)
    ax.set_xlabel('            X',fontsize=54)
    ax.set_ylabel('         Y',fontsize=54)
    #ax.text(0.05*ground_size,ground_size*0.05,s=f'k={round(1-p*0.2,1)}',color='k',fontsize=18)
    ax.set_xlim(0,ground_size)
    ax.set_ylim(0,ground_size)
    ax.xaxis.set_ticks(np.linspace(0,ground_size,3))
    ax.yaxis.set_ticks(np.linspace(0,ground_size,3))
    ax.tick_params(axis='both',which='major',labelsize=54,width=4,length=8)
    plt.show()
    fig.savefig(f'./episode={epoch+1}.png', dpi=400, bbox_inches='tight')
#Fig 3
colors = ['#d8c0c3','#4a5656','#9b394b','#002CB0','#99CDE7','#868d2a','#006600']
klabel = [0.0,0.2,0.4,0.5,0.6,0.8,1.0]

action_values = np.zeros([7,100,10,10])#k,episode,state,action,parallelcritic,actor
random_values = np.zeros([7,100,10])
#actions = np.zeros([2,100,1000,100])

for error in np.arange(7):
    print('error=',round(1-klabel[error],1))
    agt = [SACAgent(state_dim, hidden_dim, action_dim, action_bound,actor_lr, critic_lr, alpha_lr, target_entropy, agttau, gamma, device) for i in range(10)]    
    testagt = [SACAgent(state_dim, hidden_dim, action_dim, action_bound,actor_lr, critic_lr, alpha_lr, target_entropy, agttau, gamma, device) for i in range(10)]    
    for count in range(10):
        agt[count].load(f'filename/error=0.0/count={count}/',99)
    #valuebaseline = np.zeros([1000])
    for epoch in np.arange(0,100):
        for count in range(10):
            testagt[count].load(f'filename/error={round(1-klabel[error],1)}/count={count}/',epoch)
        meanvalue0 = np.zeros([1000,10,10])
        meanvalue1 = np.zeros([1000,10])
        for scount in range(1000):
            sample = np.random.randint(0,len(states))
            state = torch.from_numpy(np.array([states[sample] for i in range(100)])).float()
            action_random = torch.from_numpy(np.array((np.random.random(100)-0.5)*action_bound*2)).float().unsqueeze(1)
            for baselinecount in range(10):
                meanvalue1[scount,baselinecount] = np.mean(torch.min(agt[baselinecount].critic_1(state,action_random),agt[baselinecount].critic_2(state,action_random)).detach().numpy()[:,0])
                for samplecount in range(10):
                    action = testagt[samplecount].actor.sample(state)[0]
                    meanvalue0[scount,baselinecount,samplecount] = np.mean(torch.min(agt[baselinecount].critic_1(state,action),agt[baselinecount].critic_2(state,action)).detach().numpy()[:,0])
        random_values[error,epoch] = np.mean(meanvalue1,axis=0)
        action_values[error,epoch] = np.mean(meanvalue0,axis=0)

scaled_value = np.zeros([7,100,10,10])
std = np.std(random_values,axis=(0,1))
mean = np.mean(random_values,axis=(0,1))
for count in range(10):
    ref_values = action_values[:,:,count,:]-mean[count]
    for error in range(7):
        scaled_value[error,:,count,:] = ref_values[error,:,:]*klabel[error]+(1-klabel[error])*np.random.normal(0,std[count],(100,10))
#a = np.min(np.mean(scaled_value,axis=(2,3)))
#b = np.max(np.mean(scaled_value,axis=(2,3)))-np.min(np.mean(scaled_value,axis=(2,3)))
a = np.min(np.mean(scaled_value,axis=(2,3)))
b = np.max(np.mean(scaled_value,axis=(2,3)))-np.min(np.mean(scaled_value,axis=(2,3)))
#a = np.min(scaled_value)
#b = np.max(scaled_value)-np.min(scaled_value)
scaled_value = (scaled_value-a)/b

fig = plt.figure(figsize=(4,3))
ax = plt.subplot()
for spine in ax.spines.values():
    spine.set_linewidth(2)
for i in [0,4,1,5,2,6,3]:
    
    plt.plot([],[],color=colors[i],lw=2,label=f'k={round(klabel[i],1)}',alpha=0.8)
    #for count in range(10):
    #    plt.scatter(np.arange(0,100,10)+np.random.random(10)*0.01,np.mean(scaled_value[i,::10,:,count],axis=1),color=colors[i],s=5,alpha=0.2,zorder=1)
    plt.fill_between(np.arange(100),np.mean(scaled_value[i],axis=(1,2))-np.std(np.mean(scaled_value[i],axis=1),axis=1)/np.sqrt(10)*1.96,np.mean(scaled_value[i],axis=(1,2))+np.std(np.mean(scaled_value[i],axis=1),axis=1)/np.sqrt(10)*1.96,color=colors[i],edgecolor=None,lw=0,alpha=0.2)
    #plt.fill_between(np.arange(100),np.max(np.mean(scaled_value[i],axis=1),axis=1),np.min(np.mean(scaled_value[i],axis=1),axis=1),color=colors[i],edgecolor=None,lw=0,alpha=0.2)
    plt.plot(np.arange(100),savgol_filter(np.mean(scaled_value[i],axis=(1,2)),31,3),color=colors[i],lw=2,alpha=0.8,zorder=2)
#plt.plot([0,100],[1,1],lw=2,color='k')
plt.xlabel('Episode $e$',fontsize=18)
plt.ylabel(r'$EV_k(e)$',fontsize=18)
#plt.text(10,101,s=f'r=6,mean reward={round(np.mean(reward),3)}',color='k',fontsize=11)
plt.xlim(0,100)
plt.ylim(0,1)
#plt.yscale('log')
plt.xticks(np.linspace(0,100,6),fontsize=18)
plt.yticks(np.linspace(0,1,3),fontsize=18)
ax.minorticks_on()
ax.tick_params(axis='both',which='major',labelsize=18,width=2,length=4,direction='in')
ax.tick_params(axis='both',which='minor',labelsize=18,width=2,length=2,direction='in')
plt.legend(fontsize=8,bbox_to_anchor=(1,1.22),ncol=4)
plt.show()
fig.savefig(f'./expected_values.png', dpi=1200, bbox_inches='tight')

#Fig. 4
def Cohesion(positions):
    cr = np.zeros([NN,NN])
    for i in range(NN-1):
        for j in range(i+1,NN):
            dx,dy = (positions[:,i] - positions[:,j] + ground_size/2)%ground_size - ground_size/2
            dr = np.sqrt(dx**2+dy**2)
            cr[i][j] = cr[j][i] = np.exp(-dr)
    chs = (np.sum(cr))/NN
    return chs   

OC = np.zeros([7,10,1000])
for ip in np.arange(0,7):
    print('systemerror=',round(1-klabel[ip],1))
    for count in range(10):
        test_path = f'./systemerror={round(1-klabel[ip],1)}/{count}'
        positions = np.load(test_path+f'/trace.npy')
        for tt in range(1000):
            frame = tt*20
            OC[ip][count][tt] = Cohesion(positions[frame])
fig = plt.figure(figsize=(10,8))
ax = plt.subplot()
for spine in ax.spines.values():
    spine.set_linewidth(2)
for i in [0,4,1,5,2,6,3]:
    meanOC = np.mean(OC[i],axis=0)
    seOC = np.std(OC[i],axis=0)/np.sqrt(10)*1.96
    plt.plot(np.arange(1000),savgol_filter(meanOC,101,3),color=colors[i],lw=4,alpha=0.8,label=f'k={round(klabel[i],1)}')
    plt.fill_between(np.arange(1000),meanOC-seOC,meanOC+seOC,color=colors[i],edgecolor=None,lw=0,alpha=0.2)    
    #for count in range(10):
        #plt.scatter(np.arange(0,1000,100),OC[i,count,::100],color=colors[i],s=5,alpha=0.2)
plt.xlabel('Time',fontsize=27)
plt.ylabel(r'Cohesion  $O_C$',fontsize=27)
#plt.text(10,101,s=f'r=6,mean reward={round(np.mean(reward),3)}',color='k',fontsize=27)
plt.xlim(0,1000)
plt.ylim(-0.1,4.1)
#plt.yscale('log')
plt.xticks(np.linspace(0,1000,6),fontsize=27)
plt.yticks(np.linspace(0,4,3),fontsize=27)
plt.legend(fontsize=20,bbox_to_anchor=(1.01,1.17),ncol=4)
ax.minorticks_on()
ax.tick_params(axis='both',which='major',labelsize=27,width=2,length=4,direction='in')
ax.tick_params(axis='both',which='minor',labelsize=27,width=2,length=2,direction='in')
plt.show()
fig.savefig(f'./cohesion_time.png', dpi=400, bbox_inches='tight')


#Fig. 5
klabel = [0.0,0.2,0.4,0.5,0.6,0.8,1.0]
gr = np.zeros([7,100])
ground_size = 40
NN = int(ground_size**2*rho)
dp = np.zeros([NN,NN,1000])
for ip in np.arange(7):
    p = klabel[ip]
    for count in np.arange(10):
        positions = np.load(f'./systemerror={round(1-p,1)}/{count}/trace.npy')
        for i in np.arange(NN-1):
            for j in np.arange(i+1,NN):
                tcount = 0
                for it in np.arange(-1000,-1,10):
                    #dr[i][j][count] = np.sqrt(((positions[it][0][i]-positions[it][0][j]+0.5*ground_size)%ground_size-0.5*ground_size)**2+((positions[it][1][i]-positions[it][1][j]+0.5*ground_size)%ground_size-0.5*ground_size)**2)
                    dp[i][j][tcount+count*100] = np.linalg.norm((positions[it,:,i]-positions[it,:,j]+0.5*ground_size)%ground_size-0.5*ground_size)
                    dp[j][i][tcount+count*100] = dp[i][j][tcount+count*100]
                    tcount += 1
    dr = 0.1
    for i in np.arange(1,100):
        r = i*dr
        gr[ip][i] = len(np.where((dp.reshape(1,-1)[0]-r)*(dp.reshape(1,-1)[0]-r-dr)<0)[0])/(NN*np.pi*(2*dr*r-dr**2)*NN/ground_size**2*1000)
fig = plt.figure(figsize=(3,3))
ax = plt.subplot()
for spine in ax.spines.values():
    spine.set_linewidth(2)
x = np.arange(100)*dr
for i in range(7):
    if i==1:
        continue
    plt.plot(x,gr[i],lw=2,alpha=0.8,color=colors[i],label=f'k={round(klabel[i],1)}')
plt.plot([0,10],[1,1],lw=2,ls='--',alpha=1,color='k')
plt.xlabel(r'$r$',fontsize=18)
plt.ylabel(r'$g(r)$',fontsize=18)
#plt.text(10,101,s=f'r=6,mean reward={round(np.mean(reward),3)}',color='k',fontsize=18)
plt.xlim(0,10)
plt.ylim(0,30)
plt.xticks(np.linspace(0,10,3),fontsize=18)
plt.yticks(np.linspace(0,30,4),fontsize=18)
ax.minorticks_on()
ax.tick_params(axis='both',which='major',labelsize=18,width=2,length=4,direction='in')
ax.tick_params(axis='both',which='minor',labelsize=18,width=2,length=2,direction='in')
plt.legend(fontsize=11,handlelength=1,ncol=2)
plt.show()
fig.savefig(f'./gr_intell.png', dpi=1200, bbox_inches='tight')


from scipy.optimize import curve_fit
def f_power(x,a,b):
    return a*x**b

def f_k1(x,a):
    return a*x
path =f'./systemerror='
#NN = 360
NN = 160
NT = 20000
#MSD = np.zeros([6,5,2000])
MSD = np.zeros([7,10,2000])
#ground_size = 60
ground_size = 40
#for p in range(6):
for ip in range(7):
    for count in range(10):
        #positions = np.load(path+f'={round(p*0.2,1)}/{count}/trace.npy')
        positions = np.load(path+f'{round(1-klabel[ip],1)}/{count}/trace.npy')
        pbc = np.zeros([NT,2,NN])
        for it in range(NT-1):
            dp = positions[it+1,:,:]-positions[it,:,:]
            wrap = np.where(np.abs(dp)>=ground_size*0.5)
            pbc[it+1,:,:] = pbc[it,:,:]
            pbc[it+1,:,:][wrap] -= np.sign(dp[wrap])
        positions += pbc*ground_size
        for dt in range(2000):
            MSD[ip][count][dt] = np.mean(np.linalg.norm(positions[-15000:-3000:17]-positions[-15000+dt:-3000+dt:17],axis=1)**2,axis=(0,1))

fig = plt.figure(figsize=(3,3))
ax = plt.subplot()
for spine in ax.spines.values():
    spine.set_linewidth(2)
t = np.arange(2000)*0.05
for i in range(7):
    if i==1:
        continue
    plt.plot(t,np.mean(MSD[i],axis=0),lw=2,alpha=0.8,color=colors[i],label=f'k={round(klabel[i],1)}')

t2 = np.arange(50,2000)*0.05
a = curve_fit(f_k1,t,np.mean(MSD[0],axis=0))[0]
plt.plot(t2,f_k1(t2,a)*1.1,lw=1.5,ls='--',color='k',alpha=1)
plt.text(8,f_k1(5,a)*6,r'~$t^1$',fontsize=18)
a = curve_fit(f_k1,t,np.mean(MSD[5],axis=0))[0]
plt.plot(t2,f_k1(t2,a)/1.5,lw=1.5,ls='--',color='k',alpha=1)
plt.text(30,f_k1(10,a)/2,r'~$t^1$',fontsize=18)
t3 = np.arange(30)*0.05
plt.plot(t3,t3**2*1.2,lw=1.5,ls='--',color='k',alpha=1)
plt.text(0.3,1,r'~$t^2$',fontsize=18)
#plt.axvspan(0.1, 2, color='yellow', alpha=0.1)
#plt.axvspan(2.01, 100, color='red', alpha=0.05)
plt.xlabel('Time',fontsize=18)
plt.ylabel('MSD',fontsize=18)
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-1,1e2)
plt.ylim(1e-3,2e2)
plt.xticks([0.1,1,10,100],fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=11,handlelength=1,ncol=2,loc='lower right')
ax.minorticks_on()
ax.tick_params(axis='both',which='major',labelsize=18,width=2,length=4,direction='in')
ax.tick_params(axis='both',which='minor',labelsize=18,width=2,length=2,direction='in')
# 显示动画
plt.show()
fig.savefig(f'./MSD.png', dpi=1200, bbox_inches='tight')

q6 = np.zeros([7,10,1000])
noise=0

for ip in range(7):
    path = f'./systemerror={round(1-ip*0.2,1)},kBT=1e{noise-3}'
    q6[ip]= np.load(path+'/q6_time.npy')


fig = plt.figure(figsize=(4,3))
ax1 = plt.subplot()
for spine in ax1.spines.values():
    spine.set_linewidth(2)

seoc = np.std(np.mean(OC[:,:,800:],axis=2),axis=1)/np.sqrt(10)*1.96
meanoc = np.mean(OC[:,:,800:],axis=(1,2))
ax1.errorbar(klabel,meanoc,seoc,color=colors0[3],lw=2,alpha=0.8)
ax1.plot(klabel,np.mean(OC[:,:,800:],axis=(1,2)),color=colors0[3],lw=2,alpha=0.8)
seq6 = np.std(np.mean(q6[:,:,800:],axis=2),axis=1)/np.sqrt(10)
meanq6 = np.mean(q6[:,:,800:],axis=(1,2))
#ax1.fill_between(klabel,meanq6-seq6,meanq6+seq6,color=colors0[0],edgecolor=None,lw=0,alpha=0.2)    

ax2 = ax1.twinx()
ax2.plot([],[],color=colors0[3],lw=2,alpha=0.8,label=r'$O_C$')
ax2.errorbar(klabel,meanq6,seq6,color=colors0[0],lw=2,alpha=0.8)
ax2.plot(klabel,meanq6,color=colors0[0],lw=2,alpha=0.8,label=r'$Q_6$')
#ax2.fill_between(klabel,meanoc-seoc,meanoc+seoc,color=colors0[3],edgecolor=None,lw=0,alpha=0.2)  
ax1.set_xlabel(r'Intelligence $k$',fontsize=18)
ax2.set_ylabel(r'$Q_6$',color=colors0[0],fontsize=18)
ax1.set_ylabel(r'$O_C$',color=colors0[3],fontsize=18)
#plt.text(10,101,s=f'r=6,mean reward={round(np.mean(reward),3)}',color='k',fontsize=18)
plt.xlim(0,1)
ax2.set_ylim(0.29,0.61)
ax1.set_ylim(-0.1,4.1)
#plt.yscale('log')
plt.xticks(np.linspace(0,1,6),fontsize=18)
ax2.yaxis.set_tick_params(labelsize=18,labelcolor=colors0[0])
ax2.yaxis.set_ticks(np.linspace(0.3,0.6,4))
ax1.minorticks_on()
ax2.minorticks_on()
ax1.tick_params(axis='both',which='major',labelsize=18,width=2,length=4,direction='in')
ax1.tick_params(axis='both',which='minor',labelsize=18,width=2,length=2,direction='in')
ax2.tick_params(axis='both',which='major',labelsize=18,width=2,length=4,direction='in')
ax2.tick_params(axis='both',which='minor',labelsize=18,width=2,length=2,direction='in')
ax1.yaxis.set_tick_params(labelsize=18,labelcolor=colors0[3])
ax1.yaxis.set_ticks(np.linspace(0,4,3))
plt.legend(fontsize=11)

plt.show()
fig.savefig(f'./order_k.png', dpi=1200, bbox_inches='tight')

#Fig 6
rho = 0.1
ground_size = 40
NN = int(ground_size**2*rho)
NT = 20000
correlation = np.zeros([7,380])
for ip in range(7):
    p = klabel[ip]
    directions = np.zeros([10,NT,NN])
    for count in np.arange(10):
        velocity = np.load(f'./systemerror={round(1-p,1)}/{count}/v.npy')
        directions[count] = np.arctan2(velocity[:,1,:],velocity[:,0,:])  
    for it in range(200):
        correlation[ip,it] = np.mean(np.cos(directions[:,2000:-1:3]-directions[:,1999-it:-2-it:3]))
    for it in range(180):
        correlation[ip,200+it] = np.mean(np.cos(directions[:,2000:-1:3]-directions[:,1799-it*10:-202-it*10:3]))



fig = plt.figure(figsize=(4,3))
ax = plt.subplot()
for spine in ax.spines.values():
    spine.set_linewidth(2)
x = np.hstack([np.arange(200)*0.05,np.arange(200,2000,10)*0.05])+1
for i in range(7):
    correlation_scale = correlation[i]/correlation[i,0]
    plt.plot(x,correlation_scale,lw=3,alpha=0.8,color=colors[i],label=f'k={round(klabel[i],1)}')
plt.xlabel('            Time',fontsize=18)
plt.ylabel('Correlation',fontsize=18)
#plt.text(10,101,s=f'r=6,mean reward={round(np.mean(reward),3)}',color='k',fontsize=18)
plt.xlim(1,100)
plt.xscale('log')
plt.ylim(-0.2,1.1)
plt.xticks(fontsize=18)
plt.yticks(np.linspace(0,1,3),fontsize=18)
ax.minorticks_on()
ax.tick_params(axis='both',which='major',labelsize=18,width=2,length=4,direction='in')
ax.tick_params(axis='both',which='minor',labelsize=18,width=2,length=2,direction='in')
plt.legend(fontsize=8)
ax.tick_params(axis='x',which='major',pad=6)
plt.show()
fig.savefig(f'./correlation.png', dpi=1200, bbox_inches='tight')



distance = np.zeros([7,4,10,400])
for error in np.arange(7):
    print('systemerror=',round(1-klabel[error],1))    
    
    x_neighbor = [3,0,-3,0]
    y_neighbor = [0,3,0,-3]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',   # 使用颜色编码定义颜色
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig = plt.figure(figsize=(3,3))
    ax = plt.subplot()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    for neighbor in range(4):
        for count in range(10):
            test_path = f'./systemerror/count={count}/{round(1-klabel[error],1)}'
            test_agt = SACAgent(state_dim, hidden_dim, action_dim, action_bound,actor_lr, critic_lr, alpha_lr, target_entropy, agttau, gamma, device)
            test_agt.load(test_path,99)
            tt = 400
            positions = np.zeros([2,tt])
            direction = np.zeros(tt)
            direction[0] = np.pi*0.5
            state = np.zeros(65)
            state[0] = 1
            p = np.random.random(tt)
            for it in np.arange(tt-1):
                ksi = np.random.normal(0,np.sqrt(0.001),2)
                dx = x_neighbor[neighbor]-positions[0,it]
                dy = y_neighbor[neighbor]-positions[1,it]
                dr = np.sqrt(dx**2+dy**2)
                dtheta = np.arctan2(dy,dx)-direction[it]
                state[1] = np.cos(dtheta)*dr
                state[2] = np.sin(dtheta)*dr
                state[3] = np.cos(np.pi*0.5-direction[it])
                state[4] = np.sin(np.pi*0.5-direction[it])
                if p[it]>1-klabel[error]:
                    action = test_agt.get_action(state)
                else:
                    action = (np.random.random()-0.5)*np.pi*2
                direction[it+1] = direction[it] + action*0.05    
                positions[:,it+1] = positions[:,it] + np.array([np.cos(direction[it+1]),np.sin(direction[it+1])])*0.05
                distance[error][neighbor][count][it] = np.linalg.norm([positions[0][it]-x_neighbor[neighbor],positions[1][it]-y_neighbor[neighbor]])
            plt.plot(positions[0],positions[1],color=colors[neighbor],lw=1,ls='--',alpha=0.4,zorder=1) 
    for neighbor in range(4):
        plt.scatter(x_neighbor[neighbor],y_neighbor[neighbor],c=colors[neighbor],edgecolor='k',lw=2,s=100,zorder=2)
    plt.text(-4.5,4,f'k={round(klabel[error],1)}',fontsize=18)
    plt.xlabel('            X',fontsize=18)
    plt.ylabel('         Y',fontsize=18,labelpad=-5)
    #plt.text(10,101,s=f'r=6,mean reward={round(np.mean(reward),3)}',color='k',fontsize=18)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xticks(np.linspace(-5,5,3),fontsize=18)
    plt.yticks(np.linspace(-5,5,3),fontsize=18)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='major',labelsize=18,width=2,length=4,direction='in')
    ax.tick_params(axis='both',which='minor',labelsize=18,width=2,length=2,direction='in')
    plt.show()
    fig.savefig(f'./action2_trace_k={round(1-error*0.2,1)}.png', dpi=1200, bbox_inches='tight')

fig = plt.figure(figsize=(4,3))
ax = plt.subplot()
for spine in ax.spines.values():
    spine.set_linewidth(2)
for i in [0,4,1,5,2,6,3]:
    h=plt.hist(distance[i].reshape(1,-1)[0],range=(0,3),color=colors[i],density=True,bins=100,alpha=0.1)
    #h2=plt.hist(theta[:,-1]/np.pi*180,range=(-180,180),color=colors[i],density=True,bins=20,alpha=0)
    plt.plot(h[1][0:100]+0.5*(h[1][1]-h[1][0]),savgol_filter(h[0],53,3),lw=2,color=colors[i],ls='-',alpha=0.8,label=f'k={round(klabel[i],1)}')
    #plt.scatter(h2[1][0:20]+0.5*(h2[1][1]-h2[1][0]),h2[0],facecolors='none',color=colors0[0],s=200)
plt.xlabel('Distance',fontsize=18)
plt.ylabel('Probability',fontsize=18)
#plt.text(10,101,s=f'r=6,mean reward={round(np.mean(reward),3)}',color='k',fontsize=18)
plt.xlim(0,3)
plt.ylim(0,2.4)
#plt.yscale('log')
plt.xticks(np.linspace(0,3,7),fontsize=18)
plt.yticks(np.linspace(0,2,3),fontsize=18)
plt.legend(fontsize=8,loc='upper left',ncol=4)
ax.minorticks_on()
ax.tick_params(axis='both',which='major',labelsize=18,width=2,length=4,direction='in')
ax.tick_params(axis='both',which='minor',labelsize=18,width=2,length=2,direction='in')
ax.tick_params(axis='x',which='major',pad=6)
plt.show()
fig.savefig(f'./action_trace_pdf.png', dpi=1200, bbox_inches='tight')

#Fig.7
    

OC = np.zeros([6,5])
for ip in range(6):
    for noise in range(5):
        path = f'./systemerror={round(1-ip*0.2,1)},kBT=1e{noise-3}'
        cohesion = np.load(path+'/OC_mix_time.npy')
        OC[ip][noise] = np.mean(cohesion[:,500:])

fOC = interp2d(np.linspace(0,1,6),np.arange(5),OC.T)
xnew = np.linspace(0,1,100)
ynew = np.linspace(0,4,100)
xx,yy = np.meshgrid(xnew,ynew)
fig = plt.figure(figsize=(8,6))
ax = plt.subplot()
for spine in ax.spines.values():
    spine.set_linewidth(4)
plt.scatter(xx,yy,c=fOC(xnew,ynew),cmap='Reds',vmin=0,vmax=4,s=200,alpha=0.8)
#plt.contour(xx,yy,fitparameter[:,:,1].T,levels=[0,1],colors='black')
#plt.plot([0,1.0],[0.28,-0.07],lw=3,ls='--',color='k')
plt.xlabel(r'Intelligence $k$',fontsize=36)
#plt.ylabel(r'$\sqrt{2{\gamma}{k_B}T}/{F_{intell}}$',fontsize=24)
plt.ylabel(r'Noise amplitude $\sigma$',fontsize=36)
#plt.text(10,101,s=f'r=6,mean reward={round(np.mean(reward),3)}',color='k',fontsize=18)
plt.xlim(0,1)
plt.ylim(0,4)
plt.xticks(np.linspace(0,1,6),fontsize=36)
plt.yticks(np.arange(5),[r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$','1','10'],fontsize=36)
ax.minorticks_on()
ax.tick_params(axis='both',which='major',labelsize=36,width=4,length=8,direction='in')
ax.tick_params(axis='both',which='minor',labelsize=36,width=4,length=4,direction='in')
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=36)
cbar.ax.yaxis.set_ticks(np.linspace(0,4,3))
cbar.set_label(r'$O_C$', fontsize=36)
plt.show()
fig.savefig(f'./OC_interp.png', dpi=1200, bbox_inches='tight')




q6 = np.zeros([6,5])
for ip in range(6):
    for noise in range(5):
        path = f'./systemerror={round(1-ip*0.2,1)},kBT=1e{noise-3}'
        cohesion = np.load(path+'/q6_time.npy')
        q6[ip][noise] = np.mean(cohesion[:,500:])

fq6 = interp2d(np.linspace(0,1,6),np.arange(5),q6.T)
xnew = np.linspace(0,1,100)
ynew = np.linspace(0,4,100)
xx,yy = np.meshgrid(xnew,ynew)
fig = plt.figure(figsize=(8,6))
ax = plt.subplot()
for spine in ax.spines.values():
    spine.set_linewidth(4)
plt.scatter(xx,yy,c=fq6(xnew,ynew),cmap='Blues',vmin=0.3,vmax=0.6,s=200,alpha=0.8)
#plt.contour(xx,yy,fitparameter[:,:,1].T,levels=[0,1],colors='black')
#plt.plot([0,1.0],[0.28,-0.07],lw=3,ls='--',color='k')
plt.xlabel(r'Intelligence $k$',fontsize=36)
#plt.ylabel(r'$\sqrt{2{\gamma}{k_B}T}/{F_{intell}}$',fontsize=24)
plt.ylabel(r'Noise amplitude $\sigma$',fontsize=36)
#plt.text(10,101,s=f'r=6,mean reward={round(np.mean(reward),3)}',color='k',fontsize=18)
plt.xlim(0,1)
plt.ylim(0,4)
plt.xticks(np.linspace(0,1,6),fontsize=36)
plt.yticks(np.arange(5),[r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$','1','10'],fontsize=36)
ax.minorticks_on()
ax.tick_params(axis='both',which='major',labelsize=36,width=4,length=8,direction='in')
ax.tick_params(axis='both',which='minor',labelsize=36,width=4,length=4,direction='in')
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=36)
cbar.ax.yaxis.set_ticks(np.linspace(0.3,0.6,4))
cbar.set_label(r'$Q_6$', fontsize=36)
plt.show()
fig.savefig(f'./q6_interp.png', dpi=1200, bbox_inches='tight')




ground_size=40
density = np.zeros([6,5,10,500,25])
for error in range(6):
    for i in range(5):
        for count in range(10):
            path = f'./systemerror={round(0.2*error,1)},kBT=1e{i-3}/{count}'
            positions = np.load(path+f'/trace.npy')
            for it in range(500):
                frame = 10000+it*20
                px = (positions[frame,0]+np.random.random()*8)%ground_size
                py = (positions[frame,1]+np.random.random()*8)%ground_size
                h = plt.hist2d(px,py,bins=(5,5),range=((0,40),(0,40)))
                density[error][i][count][it] = h[0].reshape(1,-1)[0]/64




T = np.zeros(10)
T[0:5] = 10**(np.arange(5)-3.0)
T[5:10] = 10**(1.0-np.arange(5))
coexist = np.zeros([6,10])
for error in range(6):
    for i in range(5):
        hd = np.where(density[error][i]>0.06)  #boundary0.1 buffer0.04
        coexist[error][i] = np.mean(density[error][i][hd]) 
        ld = np.where(density[error][i]<0.14)
        coexist[error][9-i] = np.mean(density[error][i][ld]) 

from scipy.interpolate import interp1d

fig = plt.figure(figsize=(4,3))
ax = plt.subplot()
for spine in ax.spines.values():
    spine.set_linewidth(2)
for i in range(4):
    sep = np.where(np.abs(coexist[i]-0.097)>0.01)
    plt.scatter(coexist[i][sep],T[sep],s=60,color=colors2[5-i],marker='x',alpha=0.6)
    #fit = interp1d(coexist[i][sep],np.log(T[sep]),kind='linear')
    #xnew = np.linspace(coexist[i][0],coexist[i][-1],100)
    #ynew = savgol_filter(np.exp(fit(xnew)),53,3)
    plt.plot([],[],lw=2,alpha=0.8,color=colors2[5-i],label=f'k={round(1-i*0.2,1)}')
plt.xlabel(r'Density $\rho$',fontsize=18)
plt.ylabel(r'Noise amplitude $\sigma$',fontsize=18)
#plt.text(10,101,s=f'r=6,mean reward={round(np.mean(reward),3)}',color='k',fontsize=18)
plt.xlim(-0.02,0.32)
plt.yscale('log')
plt.ylim(1e-3/3,30)
plt.xticks(np.linspace(0,0.3,4),fontsize=18)
plt.yticks([1e-3,1e-2,1e-1,1,10],fontsize=18)
ax.minorticks_on()
ax.tick_params(axis='both',which='major',labelsize=18,width=2,length=4,direction='in')
ax.tick_params(axis='both',which='minor',labelsize=18,width=2,length=2,direction='in')
plt.legend(fontsize=8)
#ax.tick_params(axis='x',which='major',pad=6)
plt.show()
fig.savefig(f'./coexist.png', dpi=1200, bbox_inches='tight')




label =[r'$10^{1}$',r'$10^{-1}$',r'$10^{-3}$']
markers =['*','s','^']
#color = ['#172778','#5986BB','#015F39']
clabel=[0,2,4]
fig = plt.figure(figsize=(4,3))
ax = plt.subplot()
for spine in ax.spines.values():
    spine.set_linewidth(2)
#plt.scatter(xx,yy,c=cc,marker='s')
for i in range(3):
    path = f'./systemerror={round(0.0,1)},kBT=1e{1-i*2}'
    q6 = np.load(path+'/q6_time.npy')
    OC = np.load(path+'/OC_mix_time.npy')
    plt.scatter([],[],color=colors0[clabel[2-i]],label=r'$\sigma$ = '+label[i],marker=markers[i],s=120,alpha=1)
    for it in np.arange(0,1000,20):
        plt.scatter(OC[:,it],q6[:,it],s=it*0.03+10,marker=markers[i],color=colors0[clabel[2-i]],alpha=it*0.0006+0.2)
    plt.xlabel(r'Cohesion $O_C$',fontsize=18)
plt.ylabel(r'Structural order $Q_6$',fontsize=18)
plt.xlim(0,4)
plt.ylim(0.26,0.64)
plt.xticks(np.linspace(0,4,5),fontsize=18)
plt.yticks(np.linspace(0.3,0.6,4),fontsize=18)
plt.legend(fontsize=12,loc='upper left')
ax.minorticks_on()
ax.tick_params(axis='both',which='major',labelsize=18,width=2,length=4,direction='in')
ax.tick_params(axis='both',which='minor',labelsize=18,width=2,length=2,direction='in')
plt.show()
fig.savefig(f'./order_space.png', dpi=1200, bbox_inches='tight')

"""


