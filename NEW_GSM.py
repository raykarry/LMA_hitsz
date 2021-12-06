# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:26:30 2021

@author: Administrator
"""
"""
1184行 将mag全部大小转到mag_z计算霍尔电场 修改时间20211123
还有913行
此外20130211事件将上游提升至185713可行
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 17:58:33 2021

@author: Administrator
"""

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from scipy import constants as const
import math
import operator
from scipy import stats
import cdflib
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import calendar 
import matplotlib.ticker as ticker
import pandas as pd
import vg
import pylab as mpl     #import matplotlib as mpl
from scipy import interpolate  
 
#设置汉字格式
# sans-serif就是无衬线字体，是一种通用字体族。
# 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica,SimHei 中文的幼圆、隶书等等
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
'''
注意：
每次variance不同是因为取得时间间隔不同，variance定义时中间时间的通量变化
法线图同理每次不同，只能起到示意图的作用 不过可以做一下平均位置下的示意图，而不是中间时间的位置示意图
在画图时，由于是将epoch时间转化成时分秒的字符串，因此我们在x轴需要设置参数，即设置x轴每个间隔需要间隔多少数据。
我们采用int(60/(((E3*60+S3)-(E*60+S))/len(tt_list)))的逻辑为(((E3*60+S3)-(E*60+S))/len(tt_list)))是将所选时间除index，得到一个index为多少秒(k)，
为我们想要的X轴的时间间隔为60秒，60/k就可以得出在我们想要的间隔下有多少index。
电子的velocity的peer分辨率无数据，因此采用的是peef
'''
__name__ == '__main__'
###将输入时间转化为时分秒
def time_change_epoch(a,b,c,d,e,f):    
    time1=datetime(a,b,c,d,e,f)
    time1=calendar.timegm(time1.timetuple())
    return time1
##找到输入点最近的位置与索引
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]
##将时分秒转化为年月日 
def time_change(time1):
    return datetime.utcfromtimestamp(time1)
def time_change1(time1):
    timeArray = time.localtime(time1)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray) 
    return otherStyleTime
##画图时将时分秒转化为年月日,并画出法线
def plot_time_change(esa_interval):
     tt_list=[]
     for i in range(len(esa_interval)):
         zn = str(time_change(esa_interval[i]))   
         H=zn[11:13]
         M=zn[14:16]
         S=zn[17:19]
         ttt=(str(H)+':'+str(M)+':'+str(S))  
         i+=1
         tt_list.append(ttt)
     tt_list=np.array(tt_list)
     return tt_list

def cal_n(a,b):
    B_u=np.array(mag_upchoose[a])
    B_d=np.array(mag_downchoose[b])
    B_cross=np.cross(B_d,B_u)
    B_var=B_d-B_u
    B1_norm=np.linalg.norm(np.cross(B_cross, B_var))
    n1=np.cross(B_cross,B_var)/B1_norm

#第二种方法
    v_u=np.array(v_upchoose[a])
    v_d=np.array(v_downchoose[b])
    v_var=v_d-v_u
    B_cross_V=np.cross(B_u,v_var)
    B2_norm=np.linalg.norm(np.cross(B_cross_V,B_var))
    n2=np.cross(B_cross_V,B_var)/B2_norm

#第三种
    B3_cross_V=np.cross(B_d,v_var)
    B3_norm=np.linalg.norm(np.cross(B3_cross_V,B_var))
    n3=np.cross(B3_cross_V,B_var)/B3_norm

#第四种
    B4_cross_V=np.cross(B_var,v_var)
    B4_norm=np.linalg.norm(np.cross(B4_cross_V,B_var))
    n4=np.cross(B4_cross_V,B_var)/B4_norm

#第五种
    n5=v_var/(np.linalg.norm(v_var))
    
    if np.dot(n1,n2)<0:
        n1=-n1
    else:
        n1=n1
    if np.dot(n3,n2)<0:
        n3=-n3
    else:
        n3=n3
    if np.dot(n4,n2)<0:
        n4=-n4
    else:
        n4=n4
    if np.dot(n5,n2)<0:
        n5=-n5
    else:
        n5=n5

    n=(n1+n2+n3+n4+n5)/5
    n=n/np.linalg.norm(n)
#计算通量变化
    '''
    注意，这里计算通量变化是坐差值，其中最重要的是时间的选取
    '''
    
    a1=velocity_i[int((choose_fgm_upbeginidx+choose_fgm_upendidx+1)/2)]
    a2=velocity_i[int((choose_fgm_downbeginidx+choose_fgm_downendidx+1)/2)]
    k1=density_i[int((choose_fgm_upbeginidx+choose_fgm_upendidx+1)/2)]*np.dot(a1,n)
    k2=density_i[int((choose_fgm_downbeginidx+choose_fgm_downendidx+1)/2)]*np.dot(a2,n)
    # a11=velocity[int((choose_fgm_upbeginidx+choose_fgm_upendidx+1)/2)]
    # a22=velocity[int((choose_fgm_downbeginidx+choose_fgm_downendidx+10)/2)]
    # k11=density[int((choose_fgm_upbeginidx+choose_fgm_upendidx+10)/2)]*np.dot(a11,n)
    # k22=density[int((choose_fgm_downbeginidx+choose_fgm_downendidx+10)/2)]*np.dot(a22,n)
    varince=k2-k1
    # varince2=k22-k11
    # efficient=abs(varince/k1+varince2/k11)
    efficient=abs(varince/k1)
    n_z=B_var
    # n=n/np.linalg.norm(n)
    n_z=n_z/np.linalg.norm(n_z)
    
    return n,efficient,n1,n2,n3,n4,n5,k1,n_z
# B5_cross_v=np.cross(v_var,B_u)
# density_ratio=n2*v_d/(n1*v_u)
# n5_norm=np.linalg.norm(np.cross(B5_cross_v,density_ratio))
# n5=np.cross(B5_cross_v,density_ratio)/n5_norm
# n=(n1+n2+n3+n4+n5)/5
def interpol(x1,x2,x3):
    x1=np.array(x1)
    x2=np.array(x2)
    x3=np.array(x3)
    d=np.interp(x1,x2,x3)
    return d
######时间输入
# Y,M,D,H,E,S=map(int,input('上游起始时间，格式为（年 月 日 时 分 秒）：').split())
# Y1,M1,D1,H1,E1,S1=map(int,input('上游终止时间，格式为（年 月 日 时 分 秒）：').split())
# Y2,M2,D2,H2,E2,S2=map(int,input('下游起始时间，格式为（年 月 日 时 分 秒）：').split())
# Y3,M3,D3,H3,E3,S3=map(int,input('下游终止时间，格式为（年 月 日 时 分 秒）：').split())
    
# Y=2013
# M=2
# D=10
# H=16
# E=29
# S=00
# H1=16
# E1=32
# S1=00
# H2=16
# E2=33
# S2=00
# H3=16
# E3=36
# S3=00
# Y1=Y
# Y2=Y
# Y3=Y
# M1=M    
# M2=M
# M3=M
# D1=D
# D2=D
# D3=D
Y=2013
M=2
D=11
H=18
E=57
S=15
H1=18
E1=58
S1=51
H2=18
E2=59
S2=10
H3=19
E3=00
S3=39
Y1=Y
Y2=Y
Y3=Y
M1=M    
M2=M
M3=M
D1=D
D2=D
D3=D
# Y=2016
# M=5
# D=7
# H=17
# E=55
# S=43
# H1=17
# E1=58
# S1=10
# H2=17
# E2=59
# S2=45
# H3=18
# E3=1
# S3=28
# Y1=Y
# Y2=Y
# Y3=Y
# M1=M    
# M2=M
# M3=M
# D1=D
# D2=D
# D3=D

#####将输入的四个时间转化为时分秒
chooseup_begintime=time_change_epoch(Y,M,D,H,E,S)
chooseup_endtime=time_change_epoch(Y1,M1,D1,H1,E1,S1)
choosedown_begintime=time_change_epoch(Y2,M2,D2,H2,E2,S2)
choosedown_endtime=time_change_epoch(Y3,M3,D3,H3,E3,S3)
if chooseup_begintime>choosedown_endtime:
    a11=chooseup_begintime
    a22=chooseup_endtime
    a33=choosedown_begintime
    a44=choosedown_endtime
    chooseup_begintime=a44
    chooseup_endtime=a33
    choosedown_begintime=a22
    choosedown_endtime=a11
else:
    chooseup_begintime=chooseup_begintime
    chooseup_endtime=chooseup_endtime
    choosedown_begintime=choosedown_begintime
    choosedown_endtime=choosedown_endtime    
###文件读取
if M<10:
    M_1=str(0)+str(M)
else:
    M_1=str(M)
if D<10:
    D_1=str(0)+str(D)
else:
    D_1=str(D)
###fgm读取
if  D<D3:
    fgm_data=cdflib.CDF('H:\data\\themis\\thb\\l2\\fgm\\'+str(Y)+'\\thb_l2_fgm_'+str(Y)+str(M_1)+str(D)+'_v01.cdf')
    fgm_data1=cdflib.CDF('H:\data\\themis\\thb\\l2\\fgm\\'+str(Y)+'\\thb_l2_fgm_'+str(Y)+str(M_1)+str(D+1)+'_v01.cdf')
    #fgm_data=np.vstack((fgm_data,fgm_data1))
    data_time=fgm_data.varget('thb_fgs_time')
    data_time1=fgm_data1.varget('thb_fgs_time')
    data_time=np.r_[data_time,data_time1]
    mag=fgm_data.varget('thb_fgs_gsm')
    mag1=fgm_data1.varget('thb_fgs_gsm')
    mag=np.r_[mag,mag1]
    ###两个卫星esa读取
    
    esa_data=cdflib.CDF('H:\data\\themis\\thb\\l2\\esa\\'+str(Y)+'\\thb_l2_esa_'+str(Y)+str(M_1)+str(D)+'_v01.cdf')
    esa_data1=cdflib.CDF('H:\data\\themis\\thb\\l2\\esa\\'+str(Y)+'\\thb_l2_esa_'+str(Y)+str(M_1)+str(D+1)+'_v01.cdf')
    #esa_data=np.vstack((esa_data,esa_data1))
    thc_esa_data=cdflib.CDF('H:\data\\themis\\thc\\l2\\esa\\'+str(Y)+'\\thc_l2_esa_'+str(Y)+str(M_1)+str(D)+'_v01.cdf')
    thc_esa_data1=cdflib.CDF('H:\data\\themis\\thc\\l2\\esa\\'+str(Y)+'\\thc_l2_esa_'+str(Y)+str(M_1)+str(D+1)+'_v01.cdf')
    #thc_esa_data=np.vstack(thc_esa_data,thc_esa_data1)
    thc_density=thc_esa_data.varget('thc_peir_density')
    thc_density1=thc_esa_data1.varget('thc_peir_density')
    thc_density=np.r_[thc_density,thc_density1]
    esa_time=esa_data.varget('thb_peir_time')
    density=esa_data.varget('thb_peir_density')
    density_e=esa_data.varget('thb_peer_density')
    velocity=esa_data.varget('thb_peir_velocity_gsm')
    temperature_i=esa_data.varget('thb_peir_avgtemp')
    temperature_e=esa_data.varget('thb_peer_avgtemp')
    peer_time=esa_data.varget('thb_peer_time')
    peir_time=esa_data.varget('thb_peir_time')
    peef_time=esa_data.varget('thb_peef_time')
    velocity_e=esa_data.varget('thb_peef_velocity_gsm')
    esa_time1=esa_data1.varget('thb_peir_time')
    density1=esa_data1.varget('thb_peir_density')
    density_e1=esa_data1.varget('thb_peer_density')
    velocity1=esa_data1.varget('thb_peir_velocity_gsm')
    temperature_i1=esa_data1.varget('thb_peir_avgtemp')
    temperature_e1=esa_data1.varget('thb_peer_avgtemp')
    peer_time_1=esa_data1.varget('thb_peer_time')
    peir_time_1=esa_data1.varget('thb_peir_time')
    peef_time_1=esa_data1.varget('thb_peef_time')
    velocity_e_1=esa_data1.varget('thb_peef_velocity_gsm')
    esa_time=np.r_[esa_time,esa_time1]
    density_i=np.r_[density,density1]
    velocity_i=np.r_[velocity,velocity1]
    temperature_e=np.r_[temperature_e,temperature_e1]
    temperature_i=np.r_[temperature_i,temperature_i1]
    density_e=np.r_[density_e,density_e1]
    peer_time=np.r_[peer_time,peer_time_1]
    peir_time=np.r_[peir_time,peir_time_1]
    peef_time=np.r_[peef_time,peef_time_1]
    velocity_e=np.r_[velocity_e,velocity_e_1]
    
    
    ##位置读取
    
    state_data=cdflib.CDF('H:\data\\themis\\thb\\l1\\state\\'+str(Y)+'\\thb_l1_state_'+str(Y)+str(M_1)+str(D_1)+'.cdf')
    state=state_data.varget('thb_pos_gsm')
    state_time=state_data.varget('thb_state_time')
    state_data1=cdflib.CDF('H:\data\\themis\\thb\\l1\\state\\'+str(Y)+'\\thb_l1_state_'+str(Y)+str(M_1)+str(D+1)+'.cdf')
    state1=state_data1.varget('thb_pos_gsm')
    state_time1=state_data1.varget('thb_state_time')
    state=np.r_[state,state1]
    state_time=np.r_[state_time,state_time1]
    
    ##电场数据读取
    Efi_data=cdflib.CDF('H:\data\\themis\\thb\\l2\\efi\\'+str(Y)+'\\thb_l2_efi_'+str(Y)+str(M_1)+str(D_1)+'_v01.cdf')
    E_data=Efi_data.varget('thb_eff_dot0_gsm')
    E_time=Efi_data.varget('thb_eff_dot0_time')
    Efi_data1=cdflib.CDF('H:\data\\themis\\thb\\l2\\efi\\'+str(Y)+'\\thb_l2_efi_'+str(Y)+str(M_1)+str(D+1)+'_v01.cdf')
    E_data1=Efi_data1.varget('thb_eff_dot0_gsm')
    E_time1=Efi_data1.varget('thb_eff_dot0_time')
    E_data=np.r_[E_data,E_data1]
    E_time=np.r_[E_time,E_time1]
    
    
else:
    fgm_data=cdflib.CDF('H:\data\\themis\\thb\\l2\\fgm\\'+str(Y)+'\\thb_l2_fgm_'+str(Y)+str(M_1)+str(D_1)+'_v01.cdf')
    data_time=fgm_data.varget('thb_fgs_time')
    mag=fgm_data.varget('thb_fgs_gsm')
    
    ###两个卫星esa读取
    
    esa_data=cdflib.CDF('H:\data\\themis\\thb\\l2\\esa\\'+str(Y)+'\\thb_l2_esa_'+str(Y)+str(M_1)+str(D_1)+'_v01.cdf')
    thc_esa_data=cdflib.CDF('H:\data\\themis\\thc\\l2\\esa\\'+str(Y)+'\\thc_l2_esa_'+str(Y)+str(M_1)+str(D_1)+'_v01.cdf')
    thc_density=thc_esa_data.varget('thc_peir_density')
    thc_density_time=thc_esa_data.varget('thc_peir_time')
    esa_time=esa_data.varget('thb_peir_time')
    density_i=esa_data.varget('thb_peir_density')
    velocity_i=esa_data.varget('thb_peir_velocity_gsm')
    temperature_i=esa_data.varget('thb_peir_avgtemp')
    temperature_e=esa_data.varget('thb_peer_avgtemp')
    density_e=esa_data.varget('thb_peer_density')
    peer_time=esa_data.varget('thb_peer_time')
    peir_time=esa_data.varget('thb_peir_time')
    velocity_e=esa_data.varget('thb_peef_velocity_gsm')
    peef_time=esa_data.varget('thb_peef_time')
    
    ##位置读取
    
    state_data=cdflib.CDF('H:\data\\themis\\thb\\l1\\state\\'+str(Y)+'\\thb_l1_state_'+str(Y)+str(M_1)+str(D_1)+'.cdf')
    state=state_data.varget('thb_pos_sse')
    state_time=state_data.varget('thb_state_time')
    
    ###电场数据
    Efi_data=cdflib.CDF('H:\data\\themis\\thb\\l2\\efi\\'+str(Y)+'\\thb_l2_efi_'+str(Y)+str(M_1)+str(D_1)+'_v01.cdf')
    E_data=Efi_data.varget('thb_eff_dot0_gsm')
    E_time=Efi_data.varget('thb_eff_dot0_time')

###########对所有数据进行插值，插值为gsm坐标系
##Density
density_i=interpol(data_time,peir_time,density_i)
##Density_e
density_e=interpol(data_time,peer_time,density_e)
##Temperature
temperature_e=interpol(data_time,peer_time,temperature_e)
temperature_i=interpol(data_time,peir_time,temperature_i)
##thcDensity
thc_density=interpol(data_time,thc_density_time,thc_density)
##velocity_i
velocity_i_x=velocity_i[:,0]
velocity_i_y=velocity_i[:,1]
velocity_i_z=velocity_i[:,2]
velocity_i_x=interpol(data_time,peir_time,velocity_i_x)
velocity_i_y=interpol(data_time,peir_time,velocity_i_y)    
velocity_i_z=interpol(data_time,peir_time,velocity_i_z)  
velocity_i=np.vstack((velocity_i_x,velocity_i_y))
velocity_i=np.vstack((velocity_i,velocity_i_z))
velocity_i=np.transpose(velocity_i)
##velocity_e
velocity_e_x=velocity_e[:,0]
velocity_e_y=velocity_e[:,1]
velocity_e_z=velocity_e[:,2]
velocity_e_x=interpol(data_time,peef_time,velocity_e_x)
velocity_e_y=interpol(data_time,peef_time,velocity_e_y)    
velocity_e_z=interpol(data_time,peef_time,velocity_e_z) 
velocity_e=np.vstack((velocity_e_x,velocity_e_y))
velocity_e=np.vstack((velocity_e,velocity_e_z))
velocity_e=np.transpose(velocity_e)
###pos
pos_x=state[:,0]
pos_y=state[:,1]
pos_z=state[:,2]
pos_x=interpol(data_time,state_time,pos_x)
pos_y=interpol(data_time,state_time,pos_y)    
pos_z=interpol(data_time,state_time,pos_z) 
pos=np.vstack((pos_x,pos_y))
pos=np.vstack(((pos,pos_z)))
pos=np.transpose(pos)
##efi
E_data=pd.DataFrame(E_data)
E_data=E_data.interpolate(method='nearest', limit_direction='forward', axis=0)
E_data=np.array(E_data)
E_x=E_data[:,0]
E_y=E_data[:,1]
E_z=E_data[:,2]
E_x=interpol(data_time,E_time,E_x)
E_y=interpol(data_time,E_time,E_y)
E_z=interpol(data_time,E_time,E_z)
E_all=np.vstack((E_x,E_y))
E_all=np.vstack((E_all,E_z))
E_all=np.transpose(E_all)
E_all_n=E_all
##

###在文件中找到与输入时间最近的位置并读取出索引
###fgm的读取
choose_fgm_upbeginidx,choose_fgm_upbegintime=find_nearest( data_time, chooseup_begintime )
chooseup_fgm_begintime_ym=time_change(choose_fgm_upbegintime)

choose_fgm_upendidx,choose_fgm_upendtime=find_nearest( data_time, chooseup_endtime )
chooseup_fgm_endtime_ym=time_change(choose_fgm_upendtime)

choose_fgm_downbeginidx,choose_fgm_downbegintime=find_nearest( data_time, choosedown_begintime )
choosedown_fgm_begintime_ym=time_change(choose_fgm_downbegintime)

choose_fgm_downendidx,choose_fgm_downendtime=find_nearest(data_time,choosedown_endtime)
choosedown_fgm_endtime_ym=time_change(choose_fgm_downendtime)



##############数据读取
mag_upchoose=mag[choose_fgm_upbeginidx:choose_fgm_upendidx+1]
mag__timeupchoose=data_time[choose_fgm_upbeginidx:choose_fgm_upendidx+1]
mag_downchoose=mag[choose_fgm_downbeginidx:choose_fgm_downendidx+1]
mag__timedownchoose=data_time[choose_fgm_downbeginidx:choose_fgm_downendidx+1]


v_upchoose=velocity_i[choose_fgm_upbeginidx:choose_fgm_upendidx+1]
v_downchoose=velocity_i[choose_fgm_downbeginidx:choose_fgm_downendidx+1]

###计算最小方差
kk=[]
n_choose=[]
n1_choose=[]
n2_choose=[]
n3_choose=[]
n4_choose=[]
n5_choose=[]
mom_choose=[]
n_z_choose=[]


for i in range(0,len(mag_upchoose)):
    for k in range(0,len(mag_downchoose)):
        w=cal_n(i,k)      
        k+=1
        n_choose.append(w[0])
        kk.append(w[1])
        n1_choose.append(w[2])
        n2_choose.append(w[3])
        n3_choose.append(w[4])
        n4_choose.append(w[5])
        n5_choose.append(w[6])
        mom_choose.append(w[7])
        n_z_choose.append(w[8])

        if k > len(mag_downchoose)-1:
            i+=1
panduan=np.isnan(kk)
aa=[]
for i,k in enumerate(panduan):
    if k==True:
        aa.append(i)
        kk[i]=100
efficient_index,efficient=min(enumerate(kk),key=operator.itemgetter(1))
if efficient_index < len(mag_downchoose):
    k1=efficient_index
    k2=0
    n=n_choose[efficient_index]
    n1=n1_choose[efficient_index]
    n2=n2_choose[efficient_index]
    n3=n3_choose[efficient_index]
    n4=n4_choose[efficient_index]
    n5=n5_choose[efficient_index]
    mom=mom_choose[efficient_index]
    n_z=n_z_choose[efficient_index]
    print("a")
    # CTM=np.cross(n,n_t)
    print('上游所选时间:',time_change(mag__timeupchoose[int(k1)]))
    print('下游所选时间:',time_change(mag__timedownchoose[int(k2)]) )
else:
    k1=np.floor(efficient_index/len(mag_downchoose))
    k2=efficient_index-k1*len(mag_downchoose)
    n=n_choose[efficient_index]
    n1=n1_choose[efficient_index]
    n2=n2_choose[efficient_index]
    n3=n3_choose[efficient_index]
    n4=n4_choose[efficient_index]
    n5=n5_choose[efficient_index]
    n_z=n_z_choose[efficient_index]
    mom=mom_choose[efficient_index]

    # CTM=np.cross(n,n_t)
    print('上游所选时间:',time_change(mag__timeupchoose[int(k1)]))
    print('下游所选时间:',time_change(mag__timedownchoose[int(k2)]) )   
print('min:',efficient)

###########################画出法向示意图
#theta_n_B=np.arccos(np.dot(n,mag_upchoose[-1])/(np.linalg.norm(n)*np.linalg.norm(mag_upchoose[-1])))/math.pi*180
theta_n_B=vg.angle(n,mag_upchoose[-1])
n_y=np.cross(n,n_z)
n_y=n_y/np.linalg.norm(n_y)
#print(n_z)
print('夹角为:',theta_n_B)
if theta_n_B>90:
   theta_n_B=180-theta_n_B
else:
    theta_n_B=theta_n_B
if theta_n_B<45:
    print('准平行')
else:
    print('准垂直')
print('##'*50)

theta=np.linspace(0,2*np.pi,3600)
r=1638 
k=r*np.cos(theta)
s=r*np.sin(theta)

x_point=pos_x[int((choose_fgm_upbeginidx+choose_fgm_downendidx)/2)]
y_point=pos_y[int((choose_fgm_upbeginidx+choose_fgm_downendidx)/2)]
z_point=pos_z[int((choose_fgm_upbeginidx+choose_fgm_downendidx)/2)]

fig = plt.figure(figsize=(6,6), dpi=300)
axes = fig.add_subplot(111)   
x1=np.linspace(pos_x[choose_fgm_upbeginidx],pos_x[choose_fgm_downendidx],2000)
#y=(-n[0]/n[1])*x
y1=(n1[1]/n1[0])*(x1-x_point)+y_point
#plt.plot(x1,y1)
axes.plot(x1,y1,label='method_1')



#y=(-n[0]/n[1])*x
y2=(n2[1]/n2[0])*(x1-x_point)+y_point
#plt.plot(x1,y2)
axes.plot(x1,y2,label='method_2')

#y=(-n[0]/n[1])*x
y3=(n3[1]/n3[0])*(x1-x_point)+y_point
# plt.plot(x1,y3)
axes.plot(x1,y3,label='method_3')

#y=(-n[0]/n[1])*x
y4=(n4[1]/n4[0])*(x1-x_point)+y_point
# plt.plot(x1,y4)
axes.plot(x1,y4,label='method_4')

#y=(-n[0]/n[1])*x
y5=(n5[1]/n5[0])*(x1-x_point)+y_point
# plt.plot(x1,y5)
axes.plot(x1,y5,label='method_5')

y6=(n[1]/n[0])*(x1-x_point)+y_point
# plt.plot(x1,y5)
axes.plot(x1,y6,color='red',label='method_mean')
x_point_plot=pos_x[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
y_point_plot=pos_y[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
z_point_plot=pos_z[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
#plt.plot(x_point_plot,y_point_plot)
axes.plot(x_point_plot,y_point_plot,label='orbit')
#plt.figure()
#plt.quiver(x_point_plot[-1], y_point_plot[-1], x_point_plot[-1]-x_point_plot[-2], y_point_plot[-1]-y_point_plot[-2], scale_units='xy', angles='xy', scale=1)
axes.quiver(x_point_plot[-1], y_point_plot[-1], x_point_plot[-1]-x_point_plot[-20], y_point_plot[-1]-y_point_plot[-20], scale_units='xy', angles='xy', scale=1)
#plt.legend()
#plt.annotate("point",xy=(x[choose_sta_downendidx],y[choose_sta_downendidx]),xytext=((np.pi/2)+1,0.8),color="blue",weight="bold",
#arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color="b"))
# plt.plot(k,s)             ###k,s是画月球
axes.plot(k,s)
if abs(y_point)>abs(x_point):
    plt_lim=abs(y_point)+300
else:
    plt_lim=abs(x_point)+300
plt.ylim(-plt_lim,plt_lim)
plt.xlim(-plt_lim,plt_lim)
plt.legend(loc="upper left")
plt.title('Normals and orbits')
plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.show()

#####画出压缩角度示意图
fig = plt.figure(figsize=(6,6), dpi=80)
x2=np.linspace(0,pos_x[choose_fgm_upbeginidx],2000)
y_n=(n[1]/n[0])*(x1-x_point)+y_point
plt.plot(x1,y_n,label='normal')
kkk=y_point/x_point
# if n[1]/n[0] >0:
#     kkk=abs(kkk)
# else:
#     kkk=-abs(kkk)
www=(y_point-((n[1]/n[0])*x_point))/(kkk-(n[1]/n[0]))
#y_n2=(kkk)*x2
if www>0:
    x3=np.linspace(0,www+200,2000)
if www<0:
    x3=np.linspace(www-200,0,2000)
y_n2=(kkk)*x3
plt.plot(x3,y_n2,label='Point to the center of lunar')
# plt.plot(x2,y_n2,label='Point to the center of lunar')
plt.legend(loc="upper left")

plt.plot(k,s)
plt.ylim(-3000,3000)
plt.xlim(-3000,3000)
plt.title('Shock compression Angle')
plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.show()

##############法向轨道示意图
fig = plt.figure(figsize=(6,6), dpi=300)
v_point_v_up=velocity_i[int(choose_fgm_upbeginidx)]
x_point_v_up=pos_x[int(choose_fgm_upbeginidx)]
y_point_v_up=pos_y[int(choose_fgm_upbeginidx)]
x_point_v_up_plot=pos_x[choose_fgm_upbeginidx-3:choose_fgm_upendidx]
x_point_v_down_plot=pos_x[choose_fgm_downbeginidx+6:choose_fgm_downendidx+9]
v_n_up=(v_point_v_up[1]/v_point_v_up[0])*(x_point_v_up_plot-x_point_v_up)+y_point_v_up
v_n_up_1=(v_point_v_up[1]-v_point_v_up[0])/(x_point_v_up_plot[1]-x_point_v_up_plot[0])*x_point_v_up_plot+(v_point_v_up[1]*x_point_v_up_plot[0]-x_point_v_up_plot[1]*v_point_v_up[0])/(x_point_v_up_plot[1]-x_point_v_up_plot[0])

v_point_v_down=velocity_i[int(choose_fgm_downbeginidx)]
x_point_v_down=pos_x[int(choose_fgm_downbeginidx)]
y_point_v_down=pos_y[int(choose_fgm_downbeginidx)+9]
v_n_down=(v_point_v_down[1]/v_point_v_down[0])*(x_point_v_down_plot-x_point_v_down)+y_point_v_down
# plt.figure(dpi=100)
# plt.axis("equal")
# plt.plot(x_point_v_up_plot,v_n_up_1,color="black")
plt.plot(x_point_v_up_plot,v_n_up,color="red",label='upstream')
plt.plot(x_point_v_down_plot,v_n_down,color="blue",label='downstream')
plt.plot(x_point_plot,y_point_plot,label='orbit')
plt.plot(x1,y_n,label='normal')
plt.legend(loc="upper left")
plt.plot(k,s)
plt.fill_between(k,s,color='black',where=k<=0,interpolate=True)
plt.ylim(-plt_lim,plt_lim)
plt.xlim(-plt_lim,plt_lim)
plt.title('Normal orbits velocities')
plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.quiver(x_point_plot[-1], y_point_plot[-1], x_point_plot[-1]-x_point_plot[-20], y_point_plot[-1]-y_point_plot[-20], scale_units='xy', angles='xy', scale=1)
plt.show()


#############计算压缩角度
theta_n=abs(np.arctan(n[1]/n[0])/math.pi*180)
theta_circle=abs(np.arctan(y_point/x_point)/math.pi*180)
print('压缩夹角为：(负值为逆时针)',theta_n-theta_circle)


#############
vel_interval=velocity_i[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
nv_true=np.dot(vel_interval,n)
vel_up_interval=velocity_i[choose_fgm_upbeginidx:choose_fgm_upendidx+1]
vel_down_interval=velocity_i[choose_fgm_downbeginidx:choose_fgm_downendidx+1]
nv_up_true=np.dot(vel_up_interval,n)
nv_down_true=np.dot(vel_down_interval,n)
density_ideal=mom/nv_true
density_i_true=density_i[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
esa_interval=esa_time[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
dis_interval=data_time[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
##############thc作为对照组#############################################
thc_density_true=thc_density[choose_fgm_upbeginidx:choose_fgm_downendidx+1]

#################计算卫星距离月球的高度

distance=np.sqrt(x_point_plot*x_point_plot+y_point_plot*y_point_plot+z_point_plot*z_point_plot)-1738
dis_tt_list=plot_time_change(dis_interval)
plt.plot(dis_tt_list,distance)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator((int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list))))))
plt.title('Distance between THB  and lunar surface')
plt.xlabel('Time')
plt.ylabel('Distance (km)')
plt.show()

#计算平均距离
aver_distance=distance[choose_fgm_upendidx-choose_fgm_upbeginidx:choose_fgm_downbeginidx-choose_fgm_upbeginidx+1]
aver_distance_time=dis_tt_list[choose_fgm_upendidx-choose_fgm_upbeginidx:choose_fgm_downbeginidx-choose_fgm_upbeginidx+1]
plt.plot(aver_distance_time,aver_distance)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator((int(60*0.5/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list))))))
plt.title('Distance between shock  and lunar surface')
plt.xlabel('Time')
plt.ylabel('Distance (km)')
plt.show()
print('LMA距月面平均距离为：',np.mean(aver_distance))

#计算V最小以及最大时的数值（沿法向） 从上往下串
if distance[choose_fgm_upendidx-choose_fgm_upbeginidx-1]> distance[choose_fgm_downbeginidx-choose_fgm_upbeginidx]:

    
    v_upmax_index,V_interval_max=max(enumerate(abs(nv_up_true)),key=operator.itemgetter(1))
    v_downmin_index,V_interval_min=min(enumerate(abs(nv_down_true)),key=operator.itemgetter(1))
    density_downmin=density_i[[v_upmax_index+choose_fgm_upbeginidx]]
    density_e_downmin=density_e[[v_upmax_index+choose_fgm_upbeginidx]]
    print('下游最小值为：',V_interval_min)
    print('速度最小值的时间为：',time_change(data_time[v_downmin_index+choose_fgm_downbeginidx]))
    print('上游速度最大值为：',V_interval_max)
    print('速度最大值的时间为：',time_change(data_time[v_upmax_index+choose_fgm_upbeginidx]))
    print('去绝对值的上游法向坐标系速度',nv_up_true[v_upmax_index])
    print('gsm坐标系下上游v',np.linalg.norm(velocity_i[v_upmax_index+choose_fgm_upbeginidx]))
    print('v与n的夹角(n指向太阳):',np.arccos(nv_up_true[v_upmax_index]/(np.linalg.norm(velocity_i[v_upmax_index+choose_fgm_upbeginidx])
                              *np.linalg.norm(n)))/math.pi*180)
else:
    v_upmin_index,V_interval_min=min(enumerate(abs(nv_up_true)),key=operator.itemgetter(1))
    v_downmax_index,V_interval_max=max(enumerate(abs(nv_down_true)),key=operator.itemgetter(1))
    density_downmin=density_i[[v_downmax_index+choose_fgm_downbeginidx]]
    density_e_downmin=density_e[[v_upmin_index+choose_fgm_upbeginidx]]
    print('下游最大值为：',V_interval_min)
    print('速度最大值的时间为：',time_change(esa_time[v_downmax_index+choose_fgm_downbeginidx]))
    print('上游速度最小值为：',V_interval_max)
    print('速度最小值的时间为：',time_change(esa_time[v_upmin_index+choose_fgm_upbeginidx]))
    print('去绝对值的上游法向坐标系速度',nv_up_true[v_upmin_index])
    print('gsm坐标系下上游v',np.linalg.norm(velocity_i[v_downmax_index+choose_fgm_upbeginidx]))
    print('v与n的夹角:',np.arccos(nv_up_true[v_downmax_index]/(np.linalg.norm(velocity_i[v_downmax_index+choose_fgm_upbeginidx])
                              *np.linalg.norm(n)))/math.pi*180)
##########
  #计算最小值时刻的阿尔芬速
u_0=const.mu_0 
fgm_minindex,fgm_mintime=find_nearest(data_time,V_interval_min)
#       B_alfen=np.dot(mag[fgm_minindex],n)
B_alfen=np.linalg.norm(mag[fgm_minindex])
v_alfen=B_alfen*10**-12/np.sqrt((u_0*density_downmin*const.m_p*10**6))
print('ALFEN速度为:',v_alfen)
print('##'*50)  

########################################
#计算磁声数 插值 （对peer插值，插值成peir）
P=density_e*const.k*(temperature_e*11600)*10**(6)+density_i*const.k*temperature_i*11600*10**(6)
#P=density_e*temperature_e*1.602*10**(-13)+density*temperature_i*1.602*10**(-13)
density_m=density_i*10**(6)*const.m_p+density_e*10**(6)*const.m_e
V_magsonic=np.sqrt(5/3*P/density_m)/1000
V_magsonic=1/2*np.sqrt(V_magsonic**2+v_alfen**2)*(1+(1-np.sqrt(4*v_alfen**2*V_magsonic**2*np.cos(theta_n_B/180*const.pi)/(V_magsonic**2+v_alfen**2)**2)))
if distance[choose_fgm_upendidx-choose_fgm_upbeginidx-1]> distance[choose_fgm_downbeginidx-choose_fgm_upbeginidx]:
    print('上游速度最大时的磁声数',V_magsonic[v_upmax_index+choose_fgm_upbeginidx])
    print('上游速度最大时的密度',density_downmin)
    print('卫星从上游（高速）到下游（低速）')
    print('Ti为',temperature_i[v_upmax_index+choose_fgm_upbeginidx])
    print('Te为',temperature_e[v_upmax_index+choose_fgm_upbeginidx])
    
else:
    print('下游速度最大时的磁声数',V_magsonic[v_downmax_index+choose_fgm_downbeginidx])
    print('下游速度最大时的密度',density_downmin)
    print('卫星从下游（低速）到上游（高速）')
    print('Ti为',temperature_i[v_downmax_index+choose_fgm_downbeginidx])
    print('Te为',temperature_e[v_downmax_index+choose_fgm_downbeginidx])
print('B的强度为（gsm）：',B_alfen)
print('##'*50)
########################
#######计算β（对fgm进行了插值，插成和peir相同）
 
B_0=[]
# mag_inter=np.dot(mag_inter,n)
for i in range(len(mag)):
     b_0=np.linalg.norm(mag[i,:])
     #b_0=mag_inter[i]
     #b_0=(b_0*(10**(-9)))**2
     B_0.append(b_0)
     i+=1
B_0_choose=(B_0*np.array((10**(-9))))*(B_0*np.array((10**(-9))))
p_B=np.array(B_0_choose)/(2*u_0)
β=P/p_B
β_inter=β[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
plt.plot(dis_tt_list,β_inter)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.title('Beta')
plt.xlabel('Time')
plt.ylabel('Beta')
plt.show()


##作图处理 thc与thb对照图  
plt.figure(dpi=300)
plt.plot( dis_tt_list,thc_density_true,color='blue',linewidth=1.0, linestyle="-")
plt.plot( dis_tt_list,density_i_true,color='red',linewidth=1.0, linestyle="-")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.xticks()
plt.xlabel('Time')
plt.ylabel('Density'+'(cm $ \mathit{}^{-3}$'+'All Qs)')
plt.plot(dis_tt_list,density_i_true, color="red", linewidth=1.0, linestyle="-", label="thb")
plt.plot(dis_tt_list,thc_density_true, color="blue", linewidth=1.0, linestyle="-", label="thc")
#plt.plot(tt_list,density_true, color="red", linewidth=1.0, linestyle="-", label="thb")
plt.legend(loc="upper left")
plt.title('Density of THB and THC')
plt.show()
plt.plot(dis_tt_list,abs(nv_true))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.title('V across normal')
plt.xlabel('Time')
plt.ylabel('Velocity'+'(km/s)')
plt.show()

#####理想密度与实际密度对比,通量守恒
plt.figure(dpi=300)
# plt.plot( esa_interval,density_ideal,color='blue')
# plt.plot( esa_interval,density_true,color='red')
plt.plot( dis_tt_list,density_ideal,color='blue',linewidth=1.0, linestyle="-")
plt.plot( dis_tt_list,density_i_true,color='red',linewidth=1.0, linestyle="-")
#xticks=list(range(0,len(tt_list),int(len(tt_list)/5)))
#plt.xticks([tt_list[0],tt_list[int(len(tt_list)/6)],tt_list[int(len(tt_list)/3)],tt_list[int(len(tt_list)/2)],tt_list[int(len(tt_list)/3*2)],tt_list[int(len(tt_list)/6*5)],tt_list[int(len(tt_list))-1]])
#plt.xticks([tt_list[0],tt_list[int(len(tt_list)/3)-1],tt_list[int(len(tt_list)/3*2)-1],tt_list[int(len(tt_list))-1]],['16:25','16:28','16:32','16:36'])
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.xticks()
plt.xlabel('Time')
plt.ylabel('Density'+'(cm $ \mathit{}^{-3}$'+'All Qs)')
plt.plot(dis_tt_list,density_ideal, color="blue", linewidth=1.0, linestyle="-", label="ideal")
plt.plot(dis_tt_list,density_i_true, color="red", linewidth=1.0, linestyle="-", label="true")
plt.legend(loc="upper left")
plt.title('Density of ideal and true')
plt.show()
t=pd.Series(density_ideal)
p=pd.Series(density_i_true)
corr_gust=round(p.corr(t),4)
   # corr_gust=p.corr(t)
#       t1,p1_value=stats.ttest_ind(down_density_ideal, down_density_true)
print('相关系数：',corr_gust)
t,p_value=stats.ttest_ind(density_ideal, density_i_true)
print('p值为：',p_value)
print('##'*50)
##########################上游选中时间点的E与E_n

efi_up_choose_n=np.dot(E_all,n)
efi_up_choose_n=efi_up_choose_n[choose_fgm_upbeginidx:choose_fgm_upendidx+1]
efi_upnan=np.isnan(efi_up_choose_n)
efi_trynan=[]
for i,k in enumerate(efi_upnan):
    if k==True:
        efi_up_choose_n[i]=0
        efi_trynan.append(i)
        i+=1

mean_efi_up_choose_n=np.sum(efi_up_choose_n)/(len(efi_up_choose_n)-len(efi_trynan))
print('E_up_n',mean_efi_up_choose_n)
mean_efi_up_choose_n=np.sum(abs(efi_up_choose_n))/(len(efi_up_choose_n)-len(efi_trynan))
print('E_up_n_1',mean_efi_up_choose_n)

###E作图
# norm_E_i=0
# norm_E=[]
# for norm_E_i in range(len(E_data)):
#    norm_E_if=np.linalg.norm(E_data[norm_E_i,:])
#    norm_E.append(norm_E_if)
#    norm_E_i+=1
# #norm_E=np.linalg.norm(E_data)
# efi_timelist=plot_time_change(E_time)
# plt.plot(efi_timelist,norm_E)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(efi_timelist)/int((len(E_data)/24/3600*2)*4)))
# # plt.xlim(0, len(efi_timelist))
# plt.xlabel('Time')
# plt.ylabel('E(mV/m)')
# plt.title('Norm_E')
# plt.show()
# #####在N方向的
# #norm_E_i_n=0
# norm_E_n=np.dot(E_data,n)
# #norm_E=np.linalg.norm(E_data)
# efi_timelist=plot_time_change(E_time)
# plt.plot(efi_timelist,norm_E_n)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(efi_timelist)/int((len(E_data)/24/3600*2)*4)))
# #plt.xlim(0, len(efi_timelist))
# plt.xlabel('Time')
# plt.ylabel('E(mV/m)')
# plt.title('Norm_E')
# plt.show()
# Emax_index,E_interval_max=max(enumerate(abs(np.array(norm_E))),key=operator.itemgetter(1))
# print('Emax',E_interval_max)
# print('Emax_time',time_change(E_time[Emax_index]))
# # norm_E_n=np.linalg.norm(np.dot(E_data,n))
# # plt.plot(efi_timelist,norm_E_n)
# # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(50))
# # plt.xlabel('time')
# # plt.ylabel('E')
# # plt.title('norm_E_n')
# # plt.show()

###在n方向插值后的EFI选中的电场
norm_E_n=np.dot(E_all,n)
# efi_timechoose=efi_timelist[choose_efi_upbeginidx:choose_efi_downendidx+1]
efi_data_choose=norm_E_n[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
plt.plot(dis_tt_list,efi_data_choose)
# plt.plot(efi_timechoose,efi_data_choose)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(efi_data_choose)/int((len(E_data)/24/3600*2)*4)))
#plt.xlim(0, len(efi_timechoose))
plt.xlabel('Time')
plt.ylabel('E(mV/m)')
plt.title('N')

plt.show()

##########
if 'v_upmax_index' in dir():
    print('B_n（激波法向）',np.dot(mag[v_upmax_index+choose_fgm_upbeginidx],n))
else:   
    print('B_n（激波法向）',np.dot(mag[v_downmax_index+choose_fgm_upbeginidx],n))
        
        
###########计算B_n
mag_choose_all=mag[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
mag_choose_all_n=np.dot(mag_choose_all,n)
mag_ttlist=plot_time_change(data_time[choose_fgm_upbeginidx:choose_fgm_downendidx+1])
plt.plot(mag_ttlist,mag_choose_all_n)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(mag_ttlist)))))
plt.title('B_n')
plt.xlabel('Time')
plt.ylabel('B_n'+'(nT)')
plt.show()

####尝试将mag全放在Z上！2021-1123
# mag_choose_all=mag[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
# mag_choose_all_n=np.dot(mag_choose_all,n_z)
# aaa=[]
# bbb=[]
# for i in range(len(mag_choose_all)):
#     aaa.append(np.linalg.norm(mag_choose_all[i]))
# for i in range(len(mag_choose_all_n)):
#     bbb.append(np.abs(mag_choose_all_n[i]/aaa[i]))
#     print(mag_choose_all_n[i]/aaa[i])
    
    
    
    
    
    
    
######法向动量守恒,(P是插值过的跟ESA一样):有一定的误差因为都是插值，B是三分量分别插值
n_t=n_z
def cal_mom(i_density,e_density,velocity):
    mom1=[]
    i=0
    for i in range(len(i_density)):
        mom=(i_density[i]*10**(6)*const.m_p)*velocity[i]*velocity[i]*10**6
        mom1.append(mom)
        i+=1
    return mom1
e_density=density_e[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
momentum=cal_mom(density_i_true,e_density,nv_true)
p_interval=P[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
B_cal_mom_interval=mag[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
B_t_cal_mom_interval=np.dot(B_cal_mom_interval,n_t)
B_n_cal_mom_interval=np.dot(B_cal_mom_interval,n)
mag_pressure_cal_mo=(B_t_cal_mom_interval**2-B_n_cal_mom_interval**2)*10**-18/(2*u_0)
momentum_all_n=momentum+p_interval+mag_pressure_cal_mo
plt.figure(dpi=300)
# plt.ylim(0, 4.5*10**-10)
plt.plot(dis_tt_list,momentum_all_n)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.xticks()
plt.title('Norm momentum')
plt.xlabel('Time')
plt.ylabel('Momentum Flux')
plt.show()
# print('法向动量最大变化率为：',(max(momentum_all_n)-min(momentum_all_n))/np.mean(momentum_all_n))

#########切向动量守恒
velocity_t=np.dot(vel_interval,n_t)
mom_t_interval=[]
for i in range(len(vel_interval)):
    mom_t=(density_i[i]*10**(6)*const.m_p)*velocity_t[i]*nv_true[i]*10**6
    mom_t_interval.append(mom_t)
    i+=1
mag_pre_t=(B_t_cal_mom_interval*B_n_cal_mom_interval)*10**-18/u_0
momentum_all_t=mom_t_interval-mag_pre_t
plt.figure(dpi=300)
plt.ylim(0, 3*10**(-10))
plt.plot(dis_tt_list,abs(momentum_all_t))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.xticks()
plt.title('Tangential momentum')
plt.xlabel('Time')
plt.ylabel('Momentum Flux')
plt.show()
# print('切向动量最大变化率为：',(max(momentum_all_t)-min(momentum_all_t))/np.mean(momentum_all_t))

#########能量守恒
def cal_energy_v(density_i,u_n,u_t):
        energy_velocity=[]
        i=0
        for i in range(len(density_i)):
            E_v=(1/2)*(density_i[i]*10**(6)*const.m_p)*(u_n[i]*u_n[i]+u_t[i]*u_t[i])*10**9*abs(u_n[i])
            energy_velocity.append(E_v)
            i+=1
        return energy_velocity
energy_velocity=cal_energy_v(density_i_true,nv_true,velocity_t)
energy_internal=(5/3)/(5/3-1)*p_interval*nv_true*10**3
energy_magnet=(B_t_cal_mom_interval*B_t_cal_mom_interval)*(10**(-18))*(nv_true*10**3)/u_0-B_n_cal_mom_interval*10**-9*(B_t_cal_mom_interval*10**-6*velocity_t)/u_0
energy_whole=energy_velocity+abs(energy_internal)+abs(energy_magnet)
plt.plot(dis_tt_list,energy_whole)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.xticks()
plt.title('Energy')
plt.xlabel('Time')
plt.ylabel('Energy Flux')
plt.show()
print('能量最大变化率为：',(max(energy_whole)-min(energy_whole))/np.mean(energy_whole))


####
#######计算电应力与法向动量梯度的比较
E_inter=E_all[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
E_inter_norm=[]
##计算切向和法向电场强度
for E_inter_i in range(len(E_inter)):
    kkk1=np.dot(E_inter[E_inter_i,:],n)
    E_inter_norm.append(kkk1)
    E_inter_i+=1 
E_inter_norm_t=[]
for E_inter_i_t in range(len(E_inter)):
    kkk1=np.dot(E_inter[E_inter_i_t,:],n_t)
    E_inter_norm_t.append(kkk1)
    E_inter_i_t+=1 
F_E=(((np.array(E_inter_norm_t)*10**-3)**2-(np.array(E_inter_norm)*10**-3)**2)*8.854187817*10**-12)/2
#        F_E=E_inter_norm*density_true*const.e*10**6

        
##求动量的梯度与电场力对比
time_insert_ttlist=data_time[choose_fgm_upbeginidx-1]
velocity_insert_calF=velocity_i[choose_fgm_upbeginidx-1]
velocity_insert_cal_n=np.dot(n,velocity_insert_calF)
mom_insert_F=(density_i[choose_fgm_upbeginidx-1]*10**(6)*const.m_p)*velocity_insert_cal_n*velocity_insert_cal_n*10**6
tt_list_calF=np.hstack((time_insert_ttlist,esa_interval)) 
mom_cal_F=np.hstack((mom_insert_F,momentum))

F_inter_i=1
F_inter_norm=[]
for F_inter_i in range(1,len(mom_cal_F)):
    fff1=(mom_cal_F[F_inter_i]-mom_cal_F[F_inter_i-1])/(tt_list_calF[F_inter_i]-tt_list_calF[F_inter_i-1])
    F_inter_norm.append(fff1)
    F_inter_i+=1 

#plt.plot(tt_list,F_inter_norm,color="blue", linewidth=1.0, linestyle="-", label="momentum gradient")

#将电应力与动量最对比
plt.plot(dis_tt_list,F_E,color="red", linewidth=1.0, linestyle="-", label="elecrtic ")
plt.plot(dis_tt_list,momentum,color="blue", linewidth=1.0, linestyle="-", label="momentum")

plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.xticks()
plt.ylim(10**-16, 10**-9)
plt.yscale('log')#设置纵坐标以十的次幂形式展现
#        plt.yaxis.set_major_locator(ticker.LogLocator(base=100.0, numticks=5))
plt.legend(loc="upper left")
plt.title('norm momentum')
plt.xlabel('Time')
plt.ylabel('momentum')
plt.show()

####计算V*B的E与探测到的E在n方向对比对比        
E_EB= np.dot(-np.cross(velocity_i,mag),n)
E_EB=E_EB*10**-6
E_EB_choose=E_EB[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
#F_EB=E_EB*density_true*const.e*10**6
E_inter_true_compare=[]
for E_inter_i in range(len(E_inter)):
    kkk1=np.dot(E_inter[E_inter_i,:],n)
    #kkk1=np.linalg.norm(E_inter[E_inter_i,:])
    E_inter_true_compare.append(kkk1)
    E_inter_i+=1 
plt.plot(dis_tt_list,abs(np.array(E_EB_choose)),color="blue", linewidth=1.0, linestyle="-", label="V crossB")
plt.plot(dis_tt_list,abs(np.array(E_inter_true_compare)*10**-3),color="red", linewidth=1.0, linestyle="-", label="true elecrtic")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.ylim(10**-5, 10**0)
plt.yscale('log')
plt.legend(loc="upper left")
plt.title('E')
plt.xlabel('Time')
plt.ylabel('E(V/m)')
plt.show()

###将电应力加到法向动量守恒上
plt.figure(figsize=(16,8))
plt.figure(1)
ax1=plt.subplot(221)
ax1.plot(dis_tt_list,momentum_all_n)
ax1.set_title(u'old')
ax1.set_ylabel('Norm momentum')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
ax2=plt.subplot(223)
ax2.plot(dis_tt_list,momentum_all_n+ F_E)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
ax2.set_title(u'new')
ax2.set_ylabel('Norm momentum')
plt.show()

###### 计算霍尔电场

velocity_ir_e=velocity_i-velocity_e

plt.plot(dis_tt_list,velocity_ir_e[:,0][choose_fgm_upbeginidx:choose_fgm_downendidx+1],color="red", linewidth=1.0, linestyle="-", label="x")
plt.plot(dis_tt_list,velocity_ir_e[:,1][choose_fgm_upbeginidx:choose_fgm_downendidx+1],color="blue", linewidth=1.0, linestyle="-", label="y")
plt.plot(dis_tt_list,velocity_ir_e[:,2][choose_fgm_upbeginidx:choose_fgm_downendidx+1],color="black", linewidth=1.0, linestyle="-", label="z")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.legend(loc="upper left")
plt.title('relative velocity')
plt.xlabel('Time')
plt.ylabel('Velocity(Km/s)')
plt.show()
E_hall=np.cross(velocity_ir_e,mag)
E_hall_n=np.dot(E_hall,n)*10**(3-9)#3是km-m,-9是nT-T
E_hall_choose=E_hall_n[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
E_all=E_EB_choose+E_hall_choose
plt.plot(dis_tt_list,abs(E_all),color="red", linewidth=1.0, linestyle="-", label="E_ideal + Hall term")
plt.plot(dis_tt_list,abs(np.array(E_EB_choose)),color="green", linewidth=1.0, linestyle="-", label="E_ideal")
plt.plot(dis_tt_list,abs(np.array(E_inter_true_compare)*10**-3),color="purple", linewidth=1.0, linestyle="-", label="true elecrtic")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.legend(loc="upper left")
plt.title('E')
plt.ylim(10**-5, 10**0)
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('E(V/m)')
plt.show()

plt.plot(dis_tt_list,abs(E_all),color="red", linewidth=1.0, linestyle="-", label="E_ideal + Hall term")
plt.plot(dis_tt_list,abs(np.array(E_EB_choose)),color="black", linewidth=1.0, linestyle="-", label="E_ideal")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.legend(loc="upper left")
plt.title('E')
# plt.ylim(10**-5, 10**1)
# plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('E(V/m)')
plt.show()

####    计算理想电场与实际电场比较
E_EB_all= -np.cross(velocity_i,mag)*10**-6
E_EB_all_x=E_EB_all[:,2]
efi_x_inter_1=E_z*10**-3
plt.plot(esa_time[0:1000],abs(E_EB_all_x[3000:4000]),color='black',label='idea')
plt.plot(esa_time[0:1000],abs(efi_x_inter_1[3000:4000]),color='red',label='true')
plt.legend(loc="upper left")
plt.yscale('log')
plt.ylim(10**-5, 10**0)
plt.show()

######电场

E_X_n_x=np.dot(E_all_n,n)
E_X_n_y=np.dot(E_all_n,n_y)
E_X_n_z=np.dot(E_all_n,n_z)
plt.plot(dis_tt_list,E_X_n_x[choose_fgm_upbeginidx:choose_fgm_downendidx+1],color="red", linewidth=1.0, linestyle="-", label="delta vx")
plt.plot(dis_tt_list,E_X_n_y[choose_fgm_upbeginidx:choose_fgm_downendidx+1],color="green", linewidth=1.0, linestyle="-", label="delta vy")
plt.plot(dis_tt_list,E_X_n_z[choose_fgm_upbeginidx:choose_fgm_downendidx+1],color="blue", linewidth=1.0, linestyle="-", label="delta vz")
plt.show()
######带霍尔项的法向通量能量方程
v_hall=velocity_ir_e[choose_fgm_upbeginidx:choose_fgm_downendidx+1]
v_hall_n=abs(np.dot(v_hall,n))

v_hall_t=abs(np.dot(v_hall,n_t)*10)
# energy_magnet_hall=(B_t_cal_mom_interval*B_t_cal_mom_interval)*(10**(-18))*(nv_true*10**3+v_hall_n*10**3)/u_0-B_n_cal_mom_interval*10**-9*(B_t_cal_mom_interval*(10**-6)*(v_hall_t+velocity_t))/u_0
# energy_whole_hall=energy_velocity+energy_internal+energy_magnet_hall
# plt.plot(dis_tt_list,energy_whole_hall)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
# plt.xticks()
# plt.title('Energy+hall')
# plt.xlabel('Time')
# plt.ylabel('Energy Flux')
# plt.show()
# energy_whole_up=np.mean(energy_whole[0:choose_fgm_upendidx-choose_fgm_upbeginidx])
# energy_whole_down=np.mean(energy_whole[choose_fgm_downbeginidx-choose_fgm_upbeginidx:choose_fgm_downendidx-choose_fgm_upbeginidx])
# print('energy_whole up/down:',energy_whole_up/energy_whole_down)
# energy_whole_hall_up=np.mean(energy_whole_hall[0:choose_fgm_upendidx-choose_fgm_upbeginidx+1])
# energy_whole_hall_down=np.mean(energy_whole_hall[choose_fgm_downbeginidx-choose_fgm_upbeginidx:choose_fgm_downendidx-choose_fgm_upbeginidx+1])
# print('energy_whole_hall up/down:',energy_whole_hall_up/energy_whole_hall_down)
v_y=E_X_n_x[choose_fgm_upbeginidx:choose_fgm_downendidx+1]*10**3/B_t_cal_mom_interval

energy_velocity=cal_energy_v(density_i_true,nv_true,velocity_t)
energy_internal=(5/3)/(5/3-1)*p_interval*nv_true*10**3
energy_magnet=(B_t_cal_mom_interval*B_t_cal_mom_interval)*(10**(-18))*((v_y+nv_true*10**3))/u_0-B_n_cal_mom_interval*10**-9*(B_t_cal_mom_interval*10**-6*(velocity_t+v_y))/u_0
energy_whole=energy_velocity+abs(energy_internal)+abs(energy_magnet)
plt.plot(dis_tt_list,energy_whole)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.xticks()
plt.title('Energy')
plt.xlabel('Time')
plt.ylabel('Energy Flux')
plt.show()




####BZ改变mag——z了
mag_x=mag[:,0]
mag_y=mag[:,1]
mag_z=mag[:,2]
mag_z=np.dot(mag,n_t)
aaaa=[]
for i in range(len(mag_z)):
    aaaa.append(np.linalg.norm(mag[i]))
mag_z=np.array(aaaa)
E_W=(mag_z*mag_z)*(10**-18)/(2*u_0*(1.6*10**-19)*(density_i*10**6))
E_W_choose=E_W[choose_fgm_upbeginidx-1:choose_fgm_downendidx+1]
x_inter_choose=pos_x[choose_fgm_upbeginidx-1:choose_fgm_downendidx+1]
y_inter_choose=pos_y[choose_fgm_upbeginidx-1:choose_fgm_downendidx+1]
z_inter_choose=pos_z[choose_fgm_upbeginidx-1:choose_fgm_downendidx+1]

pos_inter=np.vstack((x_inter_choose,y_inter_choose))
pos_inter=np.vstack((pos_inter,z_inter_choose))
pos_inter=np.transpose(pos_inter)

pos_n=np.dot(pos_inter,n)
E_inter_n=np.dot(E_inter,n)



potential_want=[]
for i in range(len(x_inter_choose)-1): 
    potential=(E_W_choose[i+1]-E_W_choose[i])/( pos_n[i+1]-pos_n[i])*10**-3
    potential_want.append(potential)
    i+=1
plt.plot(dis_tt_list,abs(np.array(potential_want)),color='blue',label='B_Z')
plt.plot(dis_tt_list,abs(np.array(E_inter_n)*10**-3),color="red", linewidth=1.0, linestyle="-", label="true elecrtic")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.legend(loc="upper left")
plt.title('E')
# plt.ylim(10**-5, 10**1)
# plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('E(V/m)')
plt.show()

##Hall和理论
plt.figure(dpi=300)
plt.plot(dis_tt_list,abs(np.array(E_hall_choose)),color="green", linewidth=1.0, linestyle="-", label="E_ideal_hall")
plt.plot(dis_tt_list,abs(np.array(potential_want)),color='blue', linewidth=1.0, linestyle="-", label="B_Z")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.legend(loc="upper left")
plt.title('E')
# plt.ylim(10**-5, 10**1)
# plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('E(V/m)')
plt.show()

E_all_new=abs(np.array(E_EB_choose))+abs(np.array(potential_want))
plt.figure(dpi=300)
plt.plot(dis_tt_list,abs(np.array(E_inter_n)*10**-3),color="red", linewidth=1.0, linestyle="-", label="true elecrtic")
plt.plot(dis_tt_list,abs(np.array(E_all_new)),color='blue', linewidth=1.0, linestyle="-", label="B_Z+E*B")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.legend(loc="upper left")
plt.title('E')
# plt.ylim(10**-5, 10**1)
# plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('E(V/m)')
plt.show()


###
E_all_new=np.array(E_EB_choose)-np.array(potential_want)
plt.figure(dpi=300)
plt.plot(dis_tt_list,np.array(E_inter_n)*10**-3,color="red", linewidth=1.0, linestyle="-", label="true elecrtic")
plt.plot(dis_tt_list,np.array(E_all_new),color='blue', linewidth=1.0, linestyle="-", label="B_Z+E*B")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.legend(loc="upper left")
plt.title('E')
# plt.ylim(10**-5, 10**1)
# plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('E(V/m)')
plt.show()



####delta v
velocity_i_x_shock=np.dot(velocity_i,n)
velocity_i_y_shock=np.dot(velocity_i,n_y)
velocity_i_z_shock=np.dot(velocity_i,n_z)
d_velocity_i_x_shock=velocity_i_x_shock[choose_fgm_upbeginidx:choose_fgm_downendidx+1]-velocity_i_x_shock[choose_fgm_upbeginidx]
d_velocity_i_y_shock=velocity_i_y_shock[choose_fgm_upbeginidx:choose_fgm_downendidx+1]-velocity_i_y_shock[choose_fgm_upbeginidx]
d_velocity_i_z_shock=velocity_i_z_shock[choose_fgm_upbeginidx:choose_fgm_downendidx+1]-velocity_i_z_shock[choose_fgm_upbeginidx]
plt.figure(dpi=300)
plt.plot(dis_tt_list,d_velocity_i_x_shock,color="red", linewidth=1.0, linestyle="-", label="delta vx")
plt.plot(dis_tt_list,d_velocity_i_y_shock,color="green", linewidth=1.0, linestyle="-", label="delta vy")
plt.plot(dis_tt_list,d_velocity_i_z_shock,color="blue", linewidth=1.0, linestyle="-", label="delta vz")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.legend(loc="upper left")
plt.title('delta v')
# plt.ylim(10**-5, 10**1)
# plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('km/s')
plt.show()


velocity_e_x_shock=np.dot(velocity_e,n)
velocity_e_y_shock=np.dot(velocity_e,n_y)
velocity_e_z_shock=np.dot(velocity_e,n_z)
d_velocity_e_x_shock=velocity_e_x_shock[choose_fgm_upbeginidx:choose_fgm_downendidx+1]-velocity_e_x_shock[choose_fgm_upbeginidx]
d_velocity_e_y_shock=velocity_e_y_shock[choose_fgm_upbeginidx:choose_fgm_downendidx+1]-velocity_e_y_shock[choose_fgm_upbeginidx]
d_velocity_e_z_shock=velocity_e_z_shock[choose_fgm_upbeginidx:choose_fgm_downendidx+1]-velocity_e_z_shock[choose_fgm_upbeginidx]
plt.figure(dpi=300)
plt.plot(dis_tt_list,d_velocity_e_x_shock,color="red", linewidth=1.0, linestyle="-", label="delta vx")
plt.plot(dis_tt_list,d_velocity_e_y_shock,color="green", linewidth=1.0, linestyle="-", label="delta vy")
plt.plot(dis_tt_list,d_velocity_e_z_shock,color="blue", linewidth=1.0, linestyle="-", label="delta vz")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(int(60*2/(((H3*3600+E3*60+S3)-(H*3600+E*60+S))/len(dis_tt_list)))))
plt.legend(loc="upper left")
plt.title('delta v')
# plt.ylim(10**-5, 10**1)
# plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('km/s')

plt.show()

#######电势差
E_W=(mag_z*mag_z)*(10**-18)/(2*u_0*(1.6*10**-19)*(density_i*10**6))
E_W_up=E_W[choose_fgm_upendidx]
E_W_down=E_W[choose_fgm_downbeginidx]
print('电势差为：',E_W_down-E_W_up)


    

#####反射粒子
v_y=E_inter_n*10**3/B_t_cal_mom_interval
v_fan=abs(v_y)-abs(velocity_i_x_shock[choose_fgm_upbeginidx:choose_fgm_downendidx+1])-abs(np.array(potential_want)*10**6/B_t_cal_mom_interval)

plt.plot(dis_tt_list[-30::],v_fan[-30::])
plt.show()

###
 
###BS厚度
x_BS_choose=pos_x[choose_fgm_upendidx:choose_fgm_downbeginidx+1]
y_BS_choose=pos_y[choose_fgm_upendidx:choose_fgm_downbeginidx+1]
z_BS_choose=pos_z[choose_fgm_upendidx:choose_fgm_downbeginidx+1]

pos_BS_inter=np.vstack((x_BS_choose,y_BS_choose))
pos_BS_inter=np.vstack((pos_BS_inter,z_BS_choose))
pos_BSinter=np.transpose(pos_BS_inter)
pos_BS_n=np.dot(pos_BSinter,n)
d_BS=abs(pos_BS_n[-1]-pos_BS_n[1])


density_downmin=density_i[[choose_fgm_upendidx]]
density_e_downmin=density_e[[choose_fgm_upendidx]]

permittivity=8.854187817*(10**-12)
T_e=np.sqrt((density_e_downmin*10**6*const.e**2)/(permittivity*const.m_e))
inertia_e=(const.c/T_e)*10**-3
T_i=np.sqrt((density_downmin*10**6*const.e**2)/(permittivity*const.m_p))
inertia_i=(const.c/T_i)*10**-3
print('激波厚度为：',d_BS,'离子惯性长度',inertia_i,'电子惯性长度',inertia_e)

E_EFI=pd.Series(np.array(E_inter_n)*10**-3)
E_ideal=pd.Series(np.array(potential_want))
corr_E=round(E_ideal.corr(E_EFI),4)
print('EFI和理论相关系数',corr_E)