# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 19:26:44 2021

@author: Administrator
"""
import numpy as np
import cdflib
import calendar 
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import constants as const
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
        state_data=cdflib.CDF('F:\data\\themis\\thb\\l1\\state\\'+str(self.year)+'\\thb_l1_state_'+str(self.year)+str(self.month_1)+str(self.day_1)+'.cdf')
        state=state_data.varget('thb_pos_gse')
        state_time=state_data.varget('thb_state_time')
        pos_x=state[:,0]
        pos_y=state[:,1]
        pos_z=state[:,2]
        return pos_x,pos_y,pos_z,state_time

def cal(a):
    state_time=a.read_cdf()[3]
    time=a.time_change_epoch()
    array = np.asarray(time)
    idx = (np.abs(array - state_time)).argmin()
    x=a.read_cdf()[0][idx]
    y=a.read_cdf()[1][idx]
    z=a.read_cdf()[2][idx]
    return x,y,z
    
    
#地球
theta=np.linspace(0,2*const.pi,2000) 
x_e=np.cos(theta)
y_e=np.sin(theta)
plt.figure(figsize=(6,6), dpi=80)
plt.plot(x_e,y_e)
plt.rcParams['font.size'] = 20
#Bow shock
theta_b=np.linspace(0, 0.5*const.pi,1000)
r=25/(1+0.8*np.cos(theta_b))
x_b=r*np.sin(theta_b)
y_b=r*np.cos(theta_b)
plt.plot(x_b,y_b,color='green')
plt.plot(-x_b,y_b,color='green')
plt.xlabel('Y$_G$$_S$$_M$[R$_E$]')
plt.ylabel('X$_G$$_S$$_M$[R$_E$]')
# plt.ylim(-26,26)
# plt.xlim(-26,26)
a=Timeinput(2018,11,3,21,9,36)
a1=cal(a)
b=Timeinput(2017,2,27,2,13,6)
b1=cal(b)
c=Timeinput(2016,5,7,17,59,20)
c1=cal(c)
d=Timeinput(2016,5,6,17,4,54)
d1=cal(d)
e=Timeinput(2016,5,5,16,20,37)
e1=cal(e)
f=Timeinput(2013,2,11,18,59,00)
f1=cal(f)
g=Timeinput(2013,2,10,16,32,30)
g1=cal(g)
plt.plot(a1[1]/6371,a1[0]/6371,'ro')
plt.plot(b1[1]/6371,b1[0]/6371,'ro')
plt.plot(c1[1]/6371,c1[0]/6371,'ro')
plt.plot(d1[1]/6371,d1[0]/6371,'ro')
plt.plot(e1[1]/6371,e1[0]/6371,'ro')
plt.plot(f1[1]/6371,f1[0]/6371,'ro')
plt.plot(g1[1]/6371,g1[0]/6371,'ro')
plt.xlim(-50,50)
plt.plot()
plt.savefig("4.jpeg",dpi = 600)
plt.show()  
        

# y_line=np.linspace(-3,3,1200)    
# x_line=np.zeros(1200)    
# a=Timeinput(2018,11,3,21,9,36)
# a1=cal(a)
# b=Timeinput(2017,2,27,2,13,6)
# b1=cal(b)
# c=Timeinput(2016,5,7,17,59,20)
# c1=cal(c)
# d=Timeinput(2016,5,6,17,4,54)
# d1=cal(d)
# e=Timeinput(2016,5,5,16,20,37)
# e1=cal(e)
# f=Timeinput(2013,2,11,18,59,00)
# f1=cal(f)
# g=Timeinput(2013,2,10,16,32,30)
# g1=cal(g)
# theta=np.linspace(0,2*np.pi,3600)
# r=1
# k=r*np.cos(theta)
# s=r*np.sin(theta)
# plt.figure(figsize=(6,6), dpi=80)
# plt.plot(k,s, color='black')
# plt.plot(x_line,y_line,linestyle='--', color='black') 
# plt.plot(y_line,x_line,linestyle='--', color='black') 
# plt.fill_between(k,s,color='black',where=k<=0,interpolate=True)  # 在y1和y2封闭区间内填充
# plt.plot(a1[0]/1638,a1[1]/1638,'ro')
# plt.plot(b1[0]/1638,b1[1]/1638,'ro')
# plt.plot(c1[0]/1638,c1[1]/1638,'ro')
# plt.plot(d1[0]/1638,d1[1]/1638,'ro')
# plt.plot(e1[0]/1638,e1[1]/1638,'ro')
# plt.plot(f1[0]/1638,f1[1]/1638,'ro')
# plt.plot(g1[0]/1638,g1[1]/1638,'ro')
# plt.xlim(-3,3)
# plt.ylim(-3,3)
# plt.xlabel('X$_S$$_S$$_E$[R$_M$]')
# plt.ylabel('Y$_S$$_S$$_E$[R$_M$]')
# plt.show()  
# plt.figure(figsize=(6,6), dpi=80)      
# plt.plot(k,s, color='black')
# plt.plot(a1[0]/1638,a1[2]/1638,'ro')
# plt.plot(b1[0]/1638,b1[2]/1638,'ro')
# plt.plot(c1[0]/1638,c1[2]/1638,'ro')
# plt.plot(d1[0]/1638,d1[2]/1638,'ro')
# plt.plot(e1[0]/1638,e1[2]/1638,'ro')
# plt.plot(f1[0]/1638,f1[2]/1638,'ro')
# plt.plot(g1[0]/1638,g1[2]/1638,'ro')
# plt.xlim(-3,3)
# plt.ylim(-3,3)
# plt.fill_between(k,s,color='black',where=k<=0,interpolate=True) 
# plt.xlabel('X$_S$$_S$$_E$[R$_M$]')
# plt.ylabel('Z$_S$$_S$$_E$[R$_M$]')
# plt.plot(x_line,y_line,linestyle='--', color='black') 
# plt.plot(y_line,x_line,linestyle='--', color='black') 
# plt.show()  
# plt.figure(figsize=(6,6), dpi=80)      
# plt.plot(k,s, color='black')
# plt.plot(a1[1]/1638,a1[2]/1638,'ro')
# plt.plot(b1[1]/1638,b1[2]/1638,'ro')
# plt.plot(c1[1]/1638,c1[2]/1638,'ro')
# plt.plot(d1[1]/1638,d1[2]/1638,'ro')
# plt.plot(e1[1]/1638,e1[2]/1638,'ro')
# plt.plot(f1[1]/1638,f1[2]/1638,'ro')
# plt.plot(g1[1]/1638,g1[2]/1638,'ro')
# plt.xlim(-3,3)
# plt.ylim(-3,3)
# plt.xlabel('Y$_S$$_S$$_E$[R$_M$]')
# plt.ylabel('Z$_S$$_S$$_E$[R$_M$]')
# plt.plot(x_line,y_line,linestyle='--', color='black') 
# plt.plot(y_line,x_line,linestyle='--', color='black') 
# plt.show()  
###