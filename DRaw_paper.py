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

def cal(a):
    state_time=a.read_cdf()[3]
    time=a.time_change_epoch()
    array = np.asarray(time)
    idx = (np.abs(array - state_time)).argmin()
    x=a.read_cdf()[0][idx]
    y=a.read_cdf()[1][idx]
    z=a.read_cdf()[2][idx]
    return x,y,z
    
    
     
y_line=np.linspace(-3,3,1200)    
x_line=np.zeros(1200)    
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
theta=np.linspace(0,2*np.pi,3600)
r=1
k=r*np.cos(theta)
s=r*np.sin(theta)

plt.figure(figsize=(6,6), dpi=300)
plt.plot(k,s, color='black')
plt.plot(x_line,y_line,linestyle='--', color='black') 
plt.plot(y_line,x_line,linestyle='--', color='black') 
plt.fill_between(k,s,color='black',where=k<=0,interpolate=True)  # 在y1和y2封闭区间内填充
plt.plot(a1[0]/1638,a1[1]/1638,'ro')
plt.plot(b1[0]/1638,b1[1]/1638,'ro')
plt.plot(c1[0]/1638,c1[1]/1638,'ro')
plt.plot(d1[0]/1638,d1[1]/1638,'ro')
plt.plot(e1[0]/1638,e1[1]/1638,'ro')
plt.plot(f1[0]/1638,f1[1]/1638,'ro')
plt.plot(g1[0]/1638,g1[1]/1638,'ro')
plt.axis("square")
plt.xlabel('X$_S$$_S$$_E$[R$_M$]')
plt.ylabel('Y$_S$$_S$$_E$[R$_M$]')
plt.xlim(-3,3)
plt.ylim(-3,3)
# plt.invert_yaxis()
plt.show()  
plt.figure(figsize=(6,6), dpi=300)      
plt.plot(k,s, color='black')
plt.plot(a1[0]/1638,a1[2]/1638,'ro')
plt.plot(b1[0]/1638,b1[2]/1638,'ro')
plt.plot(c1[0]/1638,c1[2]/1638,'ro')
plt.plot(d1[0]/1638,d1[2]/1638,'ro')
plt.plot(e1[0]/1638,e1[2]/1638,'ro')
plt.plot(f1[0]/1638,f1[2]/1638,'ro')
plt.plot(g1[0]/1638,g1[2]/1638,'ro')
plt.axis("square")
plt.fill_between(k,s,color='black',where=k<=0,interpolate=True) 
plt.xlabel('X$_S$$_S$$_E$[R$_M$]')
plt.ylabel('Z$_S$$_S$$_E$[R$_M$]')
plt.plot(x_line,y_line,linestyle='--', color='black') 
plt.plot(y_line,x_line,linestyle='--', color='black') 
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.show()  
plt.figure(figsize=(6,6), dpi=300)      
plt.plot(k,s, color='black')
plt.plot(a1[1]/1638,a1[2]/1638,'ro')
plt.plot(b1[1]/1638,b1[2]/1638,'ro')
plt.plot(c1[1]/1638,c1[2]/1638,'ro')
plt.plot(d1[1]/1638,d1[2]/1638,'ro')
plt.plot(e1[1]/1638,e1[2]/1638,'ro')
plt.plot(f1[1]/1638,f1[2]/1638,'ro')
plt.plot(g1[1]/1638,g1[2]/1638,'ro')
plt.axis("square")
plt.xlabel('Y$_S$$_S$$_E$[R$_M$]')
plt.ylabel('Z$_S$$_S$$_E$[R$_M$]')
plt.plot(x_line,y_line,linestyle='--', color='black') 
plt.plot(y_line,x_line,linestyle='--', color='black') 
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.show()  
        



# ####画饼图
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
# fig=plt.figure()
# ax1=fig.add_subplot(1,2,1)

# label=['准平行','准垂直']#定义饼图的标签，标签是列表
# explode=[0.01,0.01]#设定各项距离圆心n个半径
# #plt.pie(values[-1,3:6],explode=explode,labels=label,autopct='%1.1f%%')#绘制饼图
# values=[6,1]
# ax1.pie(values,explode=explode,labels=label,colors = ['darkorange', 'dodgerblue'],autopct='%1.1f%%')#绘制饼图
# # ax1.title('类激波结构类型')#绘制标题
# #plt.savefig('./2018年饼图')#保存图片
# ax1.set_title('(a)类激波结构类型')

# ax2=fig.add_subplot(1,2,2)

# label=['准平行','准垂直']#定义饼图的标签，标签是列表
# explode=[0.01,0.01]#设定各项距离圆心n个半径
# #plt.pie(values[-1,3:6],explode=explode,labels=label,autopct='%1.1f%%')#绘制饼图
# values=[6,1]
# ax2.pie(values,explode=explode,labels=label,colors = ['darkorange', 'dodgerblue'],autopct='%1.1f%%')#绘制饼图
# # ax2.title('类激波结构类型')#绘制标题
# #plt.savefig('./2018年饼图')#保存图片
# ax2.set_title('(b)太阳风冲击类激波结构类型')
# plt.savefig('a',dpi=600 )
# plt.show()






