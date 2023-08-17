import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as mpl
import dataManipulation as dM
import time

scale = 1.00
speed = 0.173 #m/s

wb = '/home/tyler.harp/dev/ekkocortex/data/CSVData/simData/wb_8.csv'
m1 = '/home/tyler.harp/dev/ekkocortex/data/CSVData/simData/m_8.csv'
m2 = '/home/tyler.harp/dev/ekkocortex/data/CSVData/simData/m_15.csv'

GT = pd.read_csv(wb)
addedPoints = 5
step = 0.045

s_wb,ds_wb = dM.readSimEncoder(wb)
s_m1,ds_m1 = dM.readSimEncoder(m1)
s_m2,ds_m2 = dM.readSimEncoder(m2)
s_GT = dM.readGroundTruth(wb)
v_ref = np.ones(ds_wb.shape)
v_ref = pd.DataFrame(v_ref,columns=['time','vel'])
v_ref['time'] = s_m1['time']
v_ref['vel'] = v_ref['vel']*speed

data_list = [s_wb,ds_wb,s_m1,ds_m1,s_m2,ds_m2,v_ref,s_GT]
addition = np.zeros((addedPoints,2))
addition = pd.DataFrame(addition)
for i in range(addedPoints):
    addition.iloc[i,0] = i*step

for i in range(len(data_list)):
    data_list[i]['time'] = data_list[i]['time'] + addedPoints*step
    addition.columns = data_list[i].columns
    data_list[i] = pd.concat([addition,data_list[i]],axis=0)
    data_list[i] = data_list[i].reset_index(drop=True)
# s_ms = dM.copydfShape(s_m2,0) 
# s_ms['s'] = (s_m1['s']+s_m2['s'])/2
# s_ms['time'] = data['time']

Q_set = 0.6311111111
Rm_set = 0.4677777778
Rwb_set = 0.440555555555556
g_scale = 1.017733333

Q = np.identity(2)*Q_set
Rm = np.identity(2)*Rm_set
Rwb = np.identity(2)*Rwb_set
s_ref = dM.integrateV(v_ref)
kalman = dM.KVC(data_list[6],data_list[3],data_list[5],data_list[1],Q,Rm,Rwb,g_scale)

f1 = mpl.figure()
mpl.plot(data_list[7]['time'],data_list[7]['s'])
mpl.plot(data_list[0]['time'],data_list[0]['s'])
mpl.plot(data_list[2]['time'],data_list[2]['s']*scale)
mpl.plot(data_list[4]['time'],data_list[4]['s']*scale)
mpl.plot(kalman['time'],kalman['s'])
mpl.title('Height vs Time')
mpl.xlabel('Time (s)')
mpl.ylabel('Height (m)')
mpl.legend(['GndTrth','WB','M1','M2','Kalman'])

mpl.show()





