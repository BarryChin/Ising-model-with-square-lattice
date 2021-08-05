# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:58:49 2021

@author: 找颗星星   参考来源： https://www.guanjihuan.com/archives/1249 
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import pi
from scipy.interpolate import make_interp_spline

class ising_squre():#定义一个ising_squre的类
    def __init__(self, temp=3.0, width=7):
        self.width = width#二维正方形
        self.num_spins = width**2#总的粒子数
        L,N = self.width,self.num_spins
        self.nns = {i : (  (i//L)*L+(i+1)%L, (i+L)%N, (i//L)*L+(i-1)%L, (i-L)%N  ) for i in list(range(N))}#构造周期性边界条件
        self.config_spins = np.random.choice([-0.5,0.5],self.num_spins)*pi#构造初始态
        self.temp = temp
        self.energy = np.sum(self.get_energy())/self.num_spins/2#求单个自旋平均能量
  
    def sweep(self):
        beta = 1.0 / self.temp
        index_spins = list(range(self.num_spins))#给各个自旋编号
        #random.shuffle(index_spins)#用于将一个列表中的元素打乱，置换
        for idx in random.sample(index_spins,len(index_spins)):#one sweep in defined as N attempts of flip每一次都要尝试翻转所有自旋一次，Markov
            #k = np.random.randint(0, N - 1)#randomly choose a spin
            energy_singlebefor = -sum( np.cos(self.config_spins[idx] -self.config_spins[n] ) for n in self.nns[idx] ) 
            energy_singleafter = -sum(  np.cos(np.negative(self.config_spins[idx])-self.config_spins[n]  ) for n in self.nns[idx]  ) 
            delta_E = energy_singleafter - energy_singlebefor
            if delta_E < 0 or np.random.uniform(0.0, 1.0) < np.exp(-beta * delta_E):#判断翻转有不有效果，无论最后翻不翻转结果都视为演化一次
                self.config_spins[idx] = np.negative(self.config_spins[idx])
        
            

    def get_energy(self):#获取单个自旋周围的能量并返回一个能量的数组
        energy_single=np.zeros(np.shape(self.config_spins))
        idx = 0
        for spin in self.config_spins: #calculate energy per spin每一个自旋的能量
            energy_single[idx] = -sum(np.cos(spin - self.config_spins[n])   for n in self.nns[idx])#nearst neighbor of kth spin
            idx +=1
        return energy_single
        
    ## Let the system evolve to equilibrium state怎么判断达到平衡态？分别从最有序和最无序开始判断两者经过相同次数的翻转后是否一样？
    def equilibrate(self,max_nsweeps=int(1e4),temp=None,H=None,show_before = False,show_after=False):#演化函数
        if temp != None:
            self.temp = temp
        dic_thermal_t = {}#字典
        dic_thermal_t['energy']=[]
        #beta = 1.0/self.temp
        energy_temp = 0  #用来放演化前的能量以便进行对比
        for k in list(range(max_nsweeps)):#先让其自由演化至平衡态
            self.sweep()     
            #list_M.append(np.abs(np.sum(S)/N))
            energy = np.sum(self.get_energy())/self.num_spins/2
            dic_thermal_t['energy'] += [energy]#每演化一次得到一个能量并添加进字典
            #print( abs(energy-energy_singletemp)/abs(energy))
            if show_before  & (k%1e3 ==0):#equilibrate中的show参数控制是否输出每隔1000步的态，默认false
                print('#sweeps=%i'% (k+1))
                print('energy=%.2f'%energy)
                self.show()
            if ( ( abs(energy)==0 or abs(energy-energy_temp)/abs(energy)<1e-4 ) & (k>500) ) or k == max_nsweeps-1 :#精度1e-4，最少演化501次，最多10000次
                print('\nequilibrium state is reached at T=%.1f'%self.temp)
                print('#sweep=%i'%k)
                #print('energy=%.2f'%energy)
                self.show()
                break
            energy_temp = energy
        n1=len(dic_thermal_t['energy'])#用来标记分割平衡前后的演化能量
        
        
        #绘制能量-步数图像
        after_steps=2**13
        error_step=9
        for ki in range(int(after_steps)):
            self.sweep()
            dic_thermal_t['energy'] += [np.sum(self.get_energy())/self.num_spins/2]
            if show_after  & (ki%1e3 ==0):#equilibrate中的show参数控制是否输出每隔1000步的态，默认false
                print('#sweeps=%i'% (ki+1))
                print('energy=%.2f'%energy)
                self.show()
        x = np.arange(len(dic_thermal_t['energy']))
        y = dic_thermal_t['energy']
        x_smooth = np.linspace(x.min(), x.max(), 1300)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
        y_smooth = make_interp_spline(x, y)(x_smooth)#用于平滑函数
        #plt.plot(x_smooth, y_smooth)#绘制能量-步数图像
        plt.ylabel(r'$ E $')
        plt.xlabel('steps')
        plt.scatter(x,y)
        plt.title('T=%.2f'%self.temp)
        plt.show()
        
        
        #绘制误差图像
        equili_energy=[]#用来盛放平衡后演化能量
        for i in range(int(after_steps)):
            equili_energy.append(dic_thermal_t['energy'][n1+i])
        ni=int(after_steps/2)
        delta_energy = np.zeros(error_step)
        energy_sum=0
        error_sum=0
        for i in range(error_step):
            #d = np.zeros(ni)
            for j in range(ni):
                equili_energy[j] = 0.5*(equili_energy[2*j]+equili_energy[2*j+1])
            for ki in range(ni):
                energy_sum+=equili_energy[ki]
            for kj in range(ni):
                error_sum+=(equili_energy[kj]-energy_sum/ni)**2
            delta_energy[i]=np.sqrt(error_sum/ni/(ni-1))
            #delta_energy[i] = np.std(d)/np.sqrt(ni-1)
            ni = int(ni/2)
            energy_sum=0
            error_sum=0
        print(delta_energy)
        x1 = np.arange(1,error_step+1,1)
        y1 = delta_energy
        plt.scatter(x1,y1)
        x1_smooth = np.linspace(x1.min(), x1.max(), 100)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
        y1_smooth = make_interp_spline(x1, y1)(x1_smooth)#用于平滑函数
        plt.plot(x1_smooth, y1_smooth)#绘制误差-步数图像
        plt.ylabel('error')
        plt.xlabel('binning step')
        plt.title('T=%.2f'%self.temp)
        plt.show()
        
        self.energy = np.average(equili_energy)#系统平衡演化的能量加权平均值


    def show(self,colored=True):
        config_matrix = np.reshape(self.config_spins,(self.width,self.width))#self.list2matrix(self.config_spins)
        X, Y = np.meshgrid(np.arange(0,self.width ),np.arange(0, self.width))
        U = np.cos(config_matrix )
        V = np.sin(config_matrix )
        arrow_size=16
        plt.figure(figsize=(4,4), dpi=100)
        #plt.title('Arrows scale with plot width, not view') https://blog.csdn.net/qq_41345173/article/details/111352817
        Q = plt.quiver(X, Y, U, V, angles="uv",units='dots',scale_units="dots",scale=1.0/arrow_size,pivot="middle",\
                       edgecolors="black",headwidth=arrow_size,headlength=arrow_size,headaxislength=arrow_size,zorder=100)#https://blog.csdn.net/weixin_43718675/article/details/104589175
        #qk = plt.quiverkey(Q, 0.1, 0.1, 1, r'$spin$', labelpos='E',coordinates='figure')
        plt.title('T=%.2f'%self.temp+', #spins='+str(self.width)+'x'+str(self.width))
        plt.axis('off')
       
        #绘制方格子图
        plt.figure(figsize=(4,4), dpi=100)
        x_squre=np.zeros((self.width+1,self.width+1))
        y_squre=np.zeros((self.width+1,self.width+1))
        c_squre=np.zeros((self.width,self.width))
        for i in range(self.width+1):
            for j in range(self.width+1):
                x_squre[i][j]=j
                y_squre[i][j]=i
        #print(x_squre)
        #print(y_squre)
        for i in range(self.width):
            for j in range(self.width):
                c_squre[i][j]=self.config_spins[i*self.width+j]
        #plt.plot(x_squre.ravel(),y_squre.ravel(),"ko")这个显示图中标出矩形点
        cs=plt.pcolormesh(x_squre, y_squre,c_squre,cmap ="flag",)#"flag" or "rainbow" ,可设置最大最小值vmin = -1, vmax = +1  并在后面加上语句  plt.colorbar(cs)显示色柱
        plt.title('T=%.2f'%self.temp+', #spins='+str(self.width)+'x'+str(self.width))
        plt.axis('off')
        plt.show()


dic_thermal = {}
dic_thermal['temp']=list([0.02,2.269,4])#np.linspace返回nsteps个处于上下限中的均匀分布的值
dic_thermal['energy']=[]
#dic_thermal['Cv']=[]
initial_system=ising_squre(width=8)
#initial_system.show()
for T in dic_thermal['temp']:
    print("initial state before equilibrium at T={:}".format(T))
    initial_system.show()#展示每一个温度下开始前的初态，因为在上一个温度达到平衡后我们又让其继续演化了2**13次求能量和误差
    initial_system.equilibrate(temp=T,show_after = False)
    dic_thermal['energy'] += [initial_system.energy]
    print('energy=%.2f'%initial_system.energy)
'''
    dic_thermal['Cv'] += [initial_system.Cv]

plt.plot(dic_thermal['temp'],dic_thermal['Cv'],'.')
plt.ylabel(r'$C_v$')
plt.xlabel('T')
plt.show()
'''
plt.plot(dic_thermal['temp'],dic_thermal['energy'],'.')
plt.ylabel(r'$ E $')
plt.xlabel('T')
plt.show()
