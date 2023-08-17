import pandas as pd
import numpy as np
import math as m
from numpy.linalg import inv
import time as t



def readEncoder(df,timeStamp,encoder,t0):
    ds = getCol(df,encoder,'ds')
    t = getCol(df,timeStamp,'time')
    ds,t0 = reformat(t,ds,t0)
    disp = integrateS(ds)
    return disp,ds,t0

def readSimEncoder(file):
    data = pd.read_csv(file)
    s = data.iloc[:,[0,2]]
    s = s.rename(columns={'simulated_encoder_position':'s'})
    ds = s_to_ds(s)
    return s,ds
def readGroundTruth(file):
    data = pd.read_csv(file)
    s = data.iloc[:,[0,1]]
    s = s.rename(columns={'ground_truth_elevation':'s'})
    return s

def readVelocity(df,timeStamp,velname,t0):
    v = getCol(df,velname,'vel')
    t = getCol(df,timeStamp,'time')
    v,t0 = reformat(t,v,t0)
    return v,t0

def GetnCalRTS(df,refclock,rtsclock,rtsdata,offset):
    rtsclock = getCol(df,rtsclock,'time')
    rtsdata = getCol(df,rtsdata,'s')
    rtsdata = dropNan(rtsdata)
    rtsclock = dropNan(rtsclock)
    rtsdata = pd.concat([rtsclock,rtsdata],axis=1)
    rtsdata = timeZero(rtsdata)
    rtsdata = TimeCalibrationRTS(refclock,rtsdata,offset)

    return rtsdata

def TimeCalibrationRTS(refclock,rtsclock,offset):
    tspan_rts = rtsclock['time'].iloc[-1]
    tspan_ref = refclock['time'].iloc[-1]
    diff = abs(tspan_rts-tspan_ref)
    diffPoints= m.floor((diff+offset)/rtsclock['time'].diff().mean())
    if tspan_rts < tspan_ref: #this conditional does not work
        rtsclock = rtsclock
        # offset = pd.concat([pd.DataFrame(np.zeros([int(diffPoints),1]),columns=list(rtsclock.columns)),offset],axis=0)
    elif tspan_rts > tspan_ref:
        rtsclock = rtsclock.drop(rtsclock.index[0:diffPoints])
        rtsclock = rtsclock.reset_index(drop=True)
        rtsclock = timeZero(rtsclock)
    return rtsclock

def timeZero(df):
    df['time'] = df['time']-df['time'][0]
    return df


def reformat(times,df,t0):
    times = times.dropna()
    times = times.reset_index(drop=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = pd.concat([times,df],axis=1)
    if df['time'][0]<t0 and df['time'].iloc[-1]>t0:
        locs = len(df[(df['time'])<t0])
        df = df.drop(index=df.index[0:locs])
        df = df.reset_index(drop=True)
        df = timeZero(df)
        return df,t0
    elif df['time'][0]==t0:
        df = timeZero(df)   
        return df,t0 
    else:
        t0 = df['time'][0]
        df = timeZero(df)
        return df,t0

def integrateS(df):
    displacements = np.zeros(df.shape)
    displacements = pd.DataFrame(displacements, columns=['time','s'])
    displacements.iloc[0,0] = df.iloc[0,0]
    for i in range(1,len(df)):
        displacements.iloc[i,1] = df.iloc[i,1]+ displacements.iloc[i-1,1]
        displacements.iloc[i,0] = df.iloc[i,0]
    return displacements

def integrateV(df):
    displacements = copydfShape(df,0)
    displacements['time'] = df['time']
    displacements = displacements.rename(columns={'vel':'s'})
    dts = df['time'].diff()
    for i in range(1,len(df)):
        displacements['s'][i] = df['vel'][i-1]*dts.iloc[i] + displacements['s'][i-1]
    return displacements

def s_to_ds(s):
    ds = copydfShape(s,0)
    ds['time'] = s['time']
    ds = ds.rename(columns={'s':'ds'})
    for i in range(1,len(s)):
        ds['ds'][i] = s['s'][i] - s['s'][i-1]
    return ds

def s_to_v(s,span):
    vels = copydfShape(s,0)
    vels = vels.rename(columns={'s':'vel'})
    vels['time'] = s['time']
    dt = s['time'].diff()
    ds = s['s'].diff()
    vels['vel'] = ds/dt
    vels['vel'][0] = 0
    vels = smoothVels(vels,span)
    return vels

def smoothVels(vels,span):
    length = len(vels)
    avgVels = copydfShape(vels,0)
    avgVels['time'] = vels['time']
    vels = pd.concat([pd.DataFrame(np.zeros([int(m.ceil(span/2)),2]),columns=['time','vel']),vels],axis=0)
    vels = pd.concat([vels,pd.DataFrame(np.zeros([int(m.ceil(span/2)),2]),columns=['time','vel'])],axis=0)
    vels = vels.reset_index(drop=True)
    for i in range(m.ceil(span/2)-1,length - m.ceil(span/2) ):
        avgVels['vel'][i] = vels['vel'][i-m.ceil(span/2):i+m.ceil(span/2)].mean()
    return avgVels
 
def dropNan(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def getCol(df,colLoc,colLabel):
    col = df.rename(columns={colLoc:colLabel})[colLabel]
    return col

def copydfShape(df,num):
    if num == 0:
        shapeCopy = np.zeros(df.shape)
    elif num == 1:
                shapeCopy = np.ones(df.shape)
    shapeCopy = pd.DataFrame(shapeCopy,columns=list(df.columns))
    return shapeCopy

def frequencyCorrection(refPts, changePts):
    correction = copydfShape(refPts,0)
    for i in range(len(refPts)):
        refTime = refPts['time'][i]
        closest = changePts.iloc[(changePts['time']-refTime).abs().argsort()[:1]].astype('float')
        correction.iloc[i,:] = closest
    return correction

def applyGravComp(velocity):
    velComp = velocity.loc[velocity['vel'] < 0] 
    return velComp

def errorAnalysis(groundTruth,compData,section):
    compData = frequencyCorrection(groundTruth,compData)
    compData_err = abs(groundTruth['s']-compData['s'])
    compMax = compData['s'].max()
    gtMax = groundTruth['s'].max()
    peakErr = abs(compMax-gtMax)
    if section == 'up':
        err_mean = compData_err.iloc[0:m.floor(len(compData_err)/2)].mean()
        err_std = compData_err.iloc[0:m.floor(len(compData_err)/2)].std()
    elif section == 'down':
        err_mean = compData_err.iloc[m.floor(len(compData_err)/2):len(compData_err)].mean()
        err_std = compData_err.iloc[m.floor(len(compData_err)/2):len(compData_err)].std()
    else:
        err_mean = compData_err.mean()
        err_std = compData_err.std()
    return peakErr,err_mean,err_std

def applyTimeOffset(offsetTime,data):
    offset = copydfShape(data,0)
    avgStep = data['time'].diff().mean() 
    addPoints = offsetTime // avgStep
    offset['time'] = data['time']
    offset.iloc[:,1] = data.iloc[:,1]
    offset = pd.concat([pd.DataFrame(np.zeros([int(addPoints),2]),columns=list(data.columns)),offset],axis=0)
    offset.reset_index(drop=True)
    offset['time'].iloc[0:len(data)] = data['time']
    offset = offset.reset_index(drop=True)
    offset = offset.drop(offset.index[len(data):-1])
    return offset

def KVC(v_ref,ds_f,ds_c,ds_b,Q,Rm,Rwb,g_scale):
            motor_v_thresh = 0.01
            motor_diff_thresh = 0.01
            wb_thresh = 0.1
            df_kalman = np.zeros([len(ds_f),3])
            df_kalman = pd.DataFrame(df_kalman, columns=['time','s','vel'])
            dv = v_ref['vel'].diff()
            #x-matrix = [s v verr]
            time = ds_f['time']
            dt = time.diff()
            v_ref = v_ref['vel']

            #state = [t s v verr]
            df_kalman.iloc[:,0] = time

            Pprev = Q
            Xprev = np.matrix([[0],[v_ref[0]]]).astype('float64')
            for i in range(4,len(ds_f)):
                F = np.matrix([[1,dt[i]],
                            [0,1]])
                H = np.matrix([[1,dt[i]],
                            [0,1]])
                dv_i = dv[i-1]
                vprev = Xprev[1]
                v = vprev+dv_i
                # v = v_ref[i]
                if v_ref[i] < 0:
                    v = v * g_scale      
                Xprev[1] = v

                #prediction step
                Xpred = F*Xprev
                Ppred = F*Pprev*F.T+Q

                # set z (observer state) based on encoder values
                Xobs_f = np.matrix([[Xprev[0,0] + ds_f.iloc[i,1]],
                                    [ds_f.iloc[i,1]/dt[i]]]).astype('float64')

                Xobs_c = np.matrix([[Xprev[0,0] + ds_c.iloc[i,1]],
                                    [ds_c.iloc[i,1]/dt[i]]]).astype('float64')
                
                Xobs_m = (Xobs_c + Xobs_f)/2

                if Xobs_m.max() > 100:
                    return df_kalman
                
                Xobs_b = np.matrix([[Xprev[0,0] + ds_b.iloc[i,1]],
                                    [ds_b.iloc[i,1]/dt[i]]]).astype('float64')
                
                vwb_prev1 = ds_b.iloc[i-1,1]/dt[i-1]
                vwb_prev2 = ds_b.iloc[i-2,1]/dt[i-2]
                vwb_prev3 = ds_b.iloc[i-3,1]/dt[i-3]
                vwb_prev4 = ds_b.iloc[i-4,1]/dt[i-4]
                vmf_prev1 = ds_f.iloc[i-1,1]/dt[i-1]
                vmb_prev1 = ds_c.iloc[i-1,1]/dt[i-1]
                vmf_prev2 = ds_f.iloc[i-2,1]/dt[i-2]
                vmb_prev2 = ds_c.iloc[i-2,1]/dt[i-2]
                vmf_prev3 = ds_f.iloc[i-3,1]/dt[i-3]
                vmb_prev3 = ds_c.iloc[i-3,1]/dt[i-3]
                vmf_prev4 = ds_f.iloc[i-4,1]/dt[i-4]
                vmb_prev4 = ds_c.iloc[i-4,1]/dt[i-4]

                
                gainM = Ppred*H.T*inv(H*Ppred*H.T + Rm)
                gainWB = Ppred*H.T*inv(H*Ppred*H.T + Rwb)
                wb_avg = 0.2*Xobs_b[1]+0.2*vwb_prev1+0.2*vwb_prev2+0.2*vwb_prev3+0.2*vwb_prev4
                wb_diff = abs(v_ref[i] - wb_avg)
                mb_avg = 0.2*Xobs_c[1]+0.2*vmb_prev1+0.2*vmb_prev2+0.2*vmb_prev3+0.2*vmb_prev4
                mf_avg = 0.2*Xobs_f[1]+0.2*vmf_prev1+0.2*vmf_prev2+0.2*vmf_prev3+0.2*vmf_prev4
                mb_diff = abs(v_ref[i] - mb_avg)
                mf_diff = abs(v_ref[i] - mf_avg)
                m_diff = abs(mb_avg-mf_avg)
                #motors cannot slip if wheelie bar is slipping
                if wb_diff > wb_thresh:
                    if (mf_diff > motor_v_thresh and mb_diff > motor_v_thresh and v_ref[i] != 0) or (m_diff > motor_diff_thresh):
                        print(time[i],mf_diff,'wb & all motor slip')
                        gainM = Ppred*H.T*inv(H*Ppred*H.T + 2*Rm)
                        gainWB = Ppred*H.T*inv(H*Ppred*H.T + 10*Rwb)
                        motor_adj = gainM*(Xobs_f-Xpred)
                    else:
                        print(time[i],wb_diff,'wb slip')
                        gainWB = np.zeros((2,2))   
                        motor_adj = gainM*(Xobs_m-Xpred)
             
                
                elif m_diff > motor_diff_thresh:
                    print(time[i],m_diff,'Motor Diff')
                    motor_adj = 0

                elif mf_diff > motor_v_thresh and mb_diff < motor_v_thresh and v_ref[i] != 0:
                    print(time[i],mf_diff,'Front Motor slip')
                    motor_adj = gainM*(Xobs_c-Xpred)

                elif mf_diff < motor_v_thresh and mb_diff > motor_v_thresh and v_ref[i] != 0:
                    print(time[i],mf_diff,'Back Motor slip')
                    motor_adj = gainM*(Xobs_f-Xpred)
                
                elif mf_diff > motor_v_thresh and mb_diff > motor_v_thresh and v_ref[i] != 0:
                    print(time[i],mf_diff,'ALL Motor slip')
                    motor_adj = 0

                else:
                    motor_adj = gainM*(Xobs_m-Xpred)

                #multiply gain by difference between measured and predicted
                Xpred = Xpred + motor_adj + gainWB*(Xobs_b-Xpred)
                Ppred = Ppred - gainM*H*Ppred - gainWB*H*Ppred

                #increment current to past
                Xprev = Xpred
                Pprev = Ppred

                df_kalman.iloc[i,1:3] = Xpred.T
            return df_kalman

def applyKalman_standard(v_ref,ds_f,ds_c,ds_b,Q,Rm,Rwb):
    df_kalman = np.zeros([len(ds_f),3])
    df_kalman = pd.DataFrame(df_kalman, columns=['time','s','vel'])

    #x-matrix = [s v verr]
    time = ds_f['time']
    v_ref = v_ref['vel']

    #state = [t s v verr]
    df_kalman.iloc[:,0] = time

    Pprev = Q
    Xprev = np.matrix([[0],[v_ref[0]]]).astype('float64')
    Xobs_f = np.matrix([[0],[0]]).astype('float64')
    Xobs_c = np.matrix([[0],[0]]).astype('float64')
    Xobs_b = np.matrix([[0],[0]]).astype('float64')
    for i in range(1,len(ds_f)):
        dt = time[i] - time[i-1]
        F = np.matrix([[1,dt],
                       [0,1]])
        H = np.matrix([[1,dt],
                       [0,1]])
        v = v_ref[i]
        if v < 0:
            # v = v*1.011 # Cedar
            v = v*.98     # Monti
        Xprev[1] = v

        # set z (observer state) based on encoder values
        Xobs_f[0] = Xprev[0] + ds_f.iloc[i,1]
        Xobs_f[1] = ds_f.iloc[i,1]/dt

        Xobs_c[0] = Xprev[0] + ds_c.iloc[i,1]
        Xobs_c[1] = ds_c.iloc[i,1]/dt

        Xobs_m = (Xobs_c + Xobs_f)/2

        Xobs_b[0] = Xprev[0] + ds_b.iloc[i,1]
        Xobs_b[1] = ds_b.iloc[i,1]/dt

        #prediction step
        Xpred = F*Xprev
        Ppred = F*Pprev*F.T+Q

        #measurement step
        gainM= Ppred*H.T*inv(H*Ppred*H.T + Rm)
        gainWB = Ppred*H.T*inv(H*Ppred*H.T + Rwb)

        #multiply gain by difference between measured and predicted
        Xpred = Xpred + gainM*(Xobs_m-Xpred) + gainWB*(Xobs_b-Xpred)
        Ppred = Ppred - gainM*H*Ppred - gainWB*H*Ppred

        #increment current to past
        Xprev = Xpred
        Pprev = Ppred

        df_kalman.iloc[i,1:3] = Xpred.T
    return df_kalman
