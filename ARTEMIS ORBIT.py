# -*- coding: utf-8 -*-
"""
Created on Sun May  2 16:06:38 2021

@author: Administrator
"""

import matplotlib.ticker as ticker
import numpy as np
import cdflib
import calendar 
from datetime import datetime
import matplotlib.pyplot as plt
class Timeinput():
    def __init__(self,year,month,day,hour,minute,second):
        self.year=year
        self.month=month
        self.day=day
        self.hour=hour
        self.minute=minute
        self.second=second
    def time_change_epoch(self):    
        time1=datetime(self.year,self.month,self.day,self.hour,self.minute,self.second)
        time1=calendar.timegm(time1.timetuple())
        return time1
    def read_cdf(self):
        if self.month<10:
            self.month_1=str(0)+str(self.month)
        else:
            self.month_1=str(self.month)
        if self.day<10:
            self.day_1=str(0)+str(self.day)
        else:
            self.day_1=str(self.day)
        state_data=cdflib.CDF('H:\data\\themis\\thb\\l1\\state\\'+str(self.year)+'\\thb_l1_state_'+str(self.year)+str(self.month_1)+str(self.day_1)+'.cdf')
        state=state_data.varget('thb_pos_sse')
        state_time=state_data.varget('thb_state_time')
        pos_x=state[:,0]
        pos_y=state[:,1]
        pos_z=state[:,2]
        return pos_x,pos_y,pos_z,state_time

def cal(a,b):
    state_time=a.read_cdf()[3]
    time=a.time_change_epoch()
    array = np.asarray(time)
    idx = (np.abs(array - state_time)).argmin()
    state_time_b=b.read_cdf()[3]
    time_b=b.time_change_epoch()
    array_b = np.asarray(time_b)
    idx_b= (np.abs(array_b - state_time_b)).argmin()
    x=a.read_cdf()[0][idx:idx_b]
    y=a.read_cdf()[1][idx:idx_b]
    z=a.read_cdf()[2][idx:idx_b]
    return x,y,z
    
a=Timeinput(2013,2,11,18,50,00)
b=Timeinput(2013,2,11,19,10,00)
a1=cal(a,b)
a_1=Timeinput(2013,2,11,18,58,21)
b_1=Timeinput(2013,2,11,19,00,39)
a2=cal(a_1,b_1)
x_1=a2[0]/1638
y_1=a2[1]/1638
plt.figure(figsize=(6,6), dpi=300)
x=a1[0]/1638
y=a1[1]/1638
r=1
theta=np.linspace(0,2*np.pi,3600)
k=r*np.cos(theta)
s=r*np.sin(theta)
plt.plot(k,s)
# plt.plot(x_1,y_1,color='green')
plt.plot(x,y,color='orange')
plt.plot(x_1,y_1,color='green')
plt.arrow(x_1[-2],y_1[-2],x_1[-1]-x_1[-2],y_1[-1]-y_1[-2],head_width = 0.03,head_length = 0.05,color='black')
plt.fill_between(k,s,color='black',where=k<=0,interpolate=True)  # 在y1和y2封闭区间内填充
plt.axis("square")
plt.xlabel('X$_S$$_S$$_E$[R$_M$]',size=15)
plt.ylabel('Y$_S$$_S$$_E$[R$_M$]',size=15)
plt.show()

z=a1[2]/1638
z_1=a2[2]/1638
s=r*np.sin(theta)
plt.figure(figsize=(6,6), dpi=300)
plt.plot(k,s)

# plt.plot(x_1,z_1,color='green')
plt.plot(x,z,color='orange')
plt.plot(x_1,z_1,color='green')
plt.axis("square")
plt.fill_between(k,s,color='black',where=k<=0,interpolate=True)  # 在y1和y2封闭区间内填充
plt.arrow(x_1[-2],z_1[-2],x_1[-1]-x_1[-2],z_1[-1]-z_1[-2],head_width = 0.03,head_length = 0.05,color='red')
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.xlabel('X$_S$$_S$$_E$[R$_M$]',size=15)
plt.ylabel('Z$_S$$_S$$_E$[R$_M$]',size=15)
plt.show()