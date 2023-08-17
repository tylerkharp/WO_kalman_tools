import pandas as pd
import numpy as np
import math as m
from numpy.linalg import inv
import multiprocessing as mp
import matplotlib.pyplot as mpl
import json


class runData:
    rts = None
    Kvc = None
    KvcL = None
    KvcR = None
    # motor_diameter = 0.10795 #meters
    # wb_diameter = 0.05715 #meters
    motor_diameter = 0.3541667 #feet
    wb_diameter = 0.1875 #feet
    height_offset = 0
    def __init__(self,data_file_name,vRef_offset,GT_offset,asset,units,operator_height_offset):
        self.units = units
        self.operator_height_offset = operator_height_offset
        self.data = pd.read_csv(data_file_name)
        self.asset_type = asset
        if (asset == 'Cedar' or asset == 'Monti'):
            self.absolute_height_offset = 0
            self.global_clock = self.data.rename(columns={'__time':'time'})['time']
            self.s_wbl,self.ds_wbl,self.t0 = self.readEncoder(
                        '/sbs/wheel_odometry/CMP_WE_EE_RLS_RM08ID0009B02L2G00_LB/header/stamp',
                        '/sbs/wheel_odometry/CMP_WE_EE_RLS_RM08ID0009B02L2G00_LB/delta_s/mean',
                        0)
            self.s_wbr,self.ds_wbr,self.t0 = self.readEncoder(
                        '/sbs/wheel_odometry/CMP_WE_EE_RLS_RM08ID0009B02L2G00_RB/header/stamp',
                        '/sbs/wheel_odometry/CMP_WE_EE_RLS_RM08ID0009B02L2G00_RB/delta_s/mean',
                        self.t0
                        )
            self.s_mlf,self.ds_mlf,self.t0 = self.readEncoder(
                        '/sbs/wheel_odometry/CMP_WE_ME_MAXON_651618_LF/header/stamp',
                        '/sbs/wheel_odometry/CMP_WE_ME_MAXON_651618_LF/delta_s/mean',
                        self.t0
                        )
            self.s_mlb,self.ds_mlb,self.t0 = self.readEncoder(
                            '/sbs/wheel_odometry/CMP_WE_ME_MAXON_651618_LB/header/stamp',
                            '/sbs/wheel_odometry/CMP_WE_ME_MAXON_651618_LB/delta_s/mean',
                            self.t0
                            )
            self.s_mrf,self.ds_mrf,self.t0 = self.readEncoder(
                            '/sbs/wheel_odometry/CMP_WE_ME_MAXON_651618_RF/header/stamp',
                            '/sbs/wheel_odometry/CMP_WE_ME_MAXON_651618_RF/delta_s/mean',
                            self.t0
                            )
            self.s_mrb,self.ds_mrb,self.t0 = self.readEncoder(
                            '/sbs/wheel_odometry/CMP_WE_ME_MAXON_651618_RB/header/stamp',
                            '/sbs/wheel_odometry/CMP_WE_ME_MAXON_651618_RB/delta_s/mean',
                            self.t0
                            )
            self.v_refL,self.t0 = self.readVelocity(
                            '/sbs/drive/drive_data/left/timestamp',
                            '/sbs/drive/drive_data/left/vel_setpoint',
                            self.t0
                            )
            self.v_refR,self.t0 = self.readVelocity(
                                    '/sbs/drive/drive_data/right/timestamp',
                                    '/sbs/drive/drive_data/right/vel_setpoint',
                                    self.t0
                                    )
            self.v_refR = self.applyTimeOffset(vRef_offset,self.v_refL)
            self.v_refL = self.applyTimeOffset(vRef_offset,self.v_refR)
            self.v_ref = (self.v_refL+self.v_refR)/2
            if asset == 'Cedar':
                self.rts= self.GetnCalRTS(
                                self.s_wbl,
                                '/tf/map/odometry_1D/header/stamp',
                                '/tf/map/odometry_1D/translation/y',
                                GT_offset)
            if asset == 'Monti':
                self.rts = self.GetnCalRTS(
                                                self.s_wbl,
                                                '/triangulation_1D_odom/header/stamp',
                                                '/triangulation_1D_odom/pose/pose/position/y',
                                                GT_offset)
        elif asset == 'Terrarium':
            self.global_clock = self.data.rename(columns={'/timestamp':'time'})['time']
            self.set_absolute_height_offset()
            self.s_wbl,self.ds_wbl = self.readCounts(
                        '/timestamp',
                        '/left_drive_outputs/encoder_pos_count',
                        'wb')
            self.s_wbr,self.ds_wbr = self.readCounts(
                        '/timestamp',
                        '/right_drive_outputs/encoder_pos_count',
                        'wb')
            self.s_mlf,self.ds_mlf = self.readCounts(
                        '/timestamp',
                        '/left_drive_outputs/axis0/pos_estimate_count',
                        'mtr')
            self.s_mlb,self.ds_mlb = self.readCounts(
                            '/timestamp',
                            '/left_drive_outputs/axis1/pos_estimate_count',
                            'mtr')
            self.s_mrf,self.ds_mrf= self.readCounts(
                            '/timestamp',
                            '/right_drive_outputs/axis0/pos_estimate_count',
                            'mtr')
            self.s_mrb,self.ds_mrb = self.readCounts(
                            '/timestamp',
                            '/right_drive_outputs/axis1/pos_estimate_count',
                            'mtr')
            self.v_refL = self.readRelativeVelocity(
                            '/timestamp',
                            '/left_drive_inputs/velocity_cmd')
            self.v_refR = self.readRelativeVelocity(
                                    '/timestamp',
                                    '/right_drive_inputs/velocity_cmd')
            self.v_refR = self.applyTimeOffset(vRef_offset,self.v_refL)
            self.v_refL = self.applyTimeOffset(vRef_offset,self.v_refR)
            self.v_ref = (self.v_refL+self.v_refR)/2
            # self.apply_encoder_flip()
            # self.RC_position = self.getCol('/brain_outputs/robot_position_ft','s')
            # self.RC_position = pd.concat((self.s_wbl['time'],self.RC_position),axis=1)
    
    def getVelocities(self,smoothing):
        self.v_wbl = self.s_to_v(self.s_wbl,smoothing)
        self.v_mlf = self.s_to_v(self.s_mlf,smoothing)
        self.v_mlb = self.s_to_v(self.s_mlb,smoothing)
        self.v_wbr = self.s_to_v(self.s_wbr,smoothing)
        self.v_mrf = self.s_to_v(self.s_mrf,smoothing)
        self.v_mrb = self.s_to_v(self.s_mrb,smoothing)
        self.v_mL = (self.v_mlb['vel'] + self.v_mlf['vel'])/2
        self.v_mL = pd.concat([self.v_mlf['time'],self.v_mL],axis=1)
        self.v_mR = (self.v_mrb['vel'] + self.v_mrf['vel'])/2
        self.v_mR = pd.concat([self.v_mrf['time'],self.v_mR],axis=1)
        if self.rts is not None:
            self.v_GT = self.s_to_v(self.rts,smoothing/2)

    def getAccelerations(self,smoothing):
        self.a_mL = (self.a_mlb['accel'] + self.a_mlf['accel'])/2
        self.a_mL = pd.concat([self.a_mlf['time'],self.a_mL],axis=1)
        self.a_mR = (self.a_mrb['accel'] + self.a_mrf['accel'])/2
        self.a_mR = pd.concat([self.a_mrf['time'],self.a_mR],axis=1)
        if self.rts is not None:
            self.a_GT = self.v_to_a(self.v_GT,smoothing/2)
        self.a_wbl = self.v_to_a(self.v_wbl,smoothing)
        self.a_mlf = self.v_to_a(self.v_mlf,smoothing)
        self.a_mlb = self.v_to_a(self.v_mlb,smoothing)
        self.a_wbr = self.v_to_a(self.v_wbr,smoothing)
        self.a_mrf = self.v_to_a(self.v_mrf,smoothing)
        self.a_mrb = self.v_to_a(self.v_mrb,smoothing)

    def readCounts(self,timeStamp,counts,encoder_type):
        if encoder_type == 'wb':
            cpr = 512
            wheel_diam = self.wb_diameter
            gear_ratio = 1
        if encoder_type == 'mtr':
            cpr = 48
            wheel_diam = self.motor_diameter #meters
            gear_ratio = 100
        should_flip = False
        if counts == '/right_drive_outputs/axis1/pos_estimate_count' or counts == '/right_drive_outputs/axis0/pos_estimate_count':
            should_flip = True
        counts = self.getCol(counts,'counts')
        time = self.getCol(timeStamp,'time')
        counts = pd.concat([time,counts],axis=1)            
        ds = self.copydfShape(counts,0)
        ds = ds.rename(columns={'counts':'ds'})
        ds['time'] = counts['time']
        count_diffs = counts['counts'].diff()
        theta = count_diffs / cpr * 2 * m.pi
        ds['ds'] = (theta * (wheel_diam/2)) / gear_ratio
        ds['ds'][0] = 0
        if should_flip:
            ds['ds'] = -ds['ds']
        disp = self.integrateS(ds)
        disp = self.timeZero(disp)
        disp['s'] = disp['s'] + self.absolute_height_offset
        ds = self.timeZero(ds)
        return disp,ds
    
    def set_absolute_height_offset(self):
        if self.operator_height_offset != 0:
            self.absolute_height_offset = self.data['/brain_outputs/robot_position_ft'].iloc[0] - self.operator_height_offset
        else:
            self.absolute_height_offset = 0
        
    def getCol(self,colLoc,colLabel):
        col = self.data.rename(columns={colLoc:colLabel})[colLabel]
        return col
    
    def readEncoder(self,timeStamp,encoder,t0):
        ds = self.getCol(encoder,'ds')
        t = self.getCol(timeStamp,'time')
        ds,t0 = self.reformat(t,ds,t0)
        disp = self.integrateS(ds)
        disp['s'] = disp['s'] + self.height_offset
        return disp,ds,t0

    def readSimEncoder(self,file):
        data = pd.read_csv(file)
        s = data.iloc[:,[0,2]]
        s = s.rename(columns={'simulated_encoder_position':'s'})
        ds = self.s_to_ds(s)
        return s,ds

    def readVelocity(self,timeStamp,velname,t0):
        v = self.getCol(velname,'vel')
        time = self.getCol(timeStamp,'time')
        v,t0 = self.reformat(time,v,t0)
        return v,t0
    
    def readRelativeVelocity(self,timeStamp,relativeVel):
        rel_v = self.getCol(relativeVel,'vel')
        time = self.getCol(timeStamp,'time')
        rel_v = pd.concat([time,rel_v],axis=1)
        vel = self.copydfShape(rel_v,0)
        vel['time'] = rel_v['time']
        vel['vel'] = rel_v['vel'].apply(self.relative_velocity_to_mps)
        vel = self.timeZero(vel)
        return vel
    
    def relative_velocity_to_mps(self,rel_v):
        max_cps = 2500 #ticks per second
        min_cps = 100 #ticks per second
        gear_ratio = 100
        cpr = 48
        wheel_radius = self.motor_diameter/2.0 #meters
        motor_count = 0
        if rel_v > 0:
            motor_count = ((max_cps-min_cps) * rel_v) / 100 + min_cps
        elif rel_v < 0:
            motor_count = ((max_cps-min_cps) * rel_v) / 100 - min_cps
        mps = 2 * m.pi * (motor_count / cpr / gear_ratio) * wheel_radius
        return mps
    
    def GetnCalRTS(self,refclock,rtsclock,rtsdata,offset):
        rtsclock = self.getCol(rtsclock,'time')
        rtsdata = self.getCol(rtsdata,'s')
        rtsdata = self.dropNan(rtsdata)
        rtsclock = self.dropNan(rtsclock)
        rtsdata = pd.concat([rtsclock,rtsdata],axis=1)
        rtsdata = self.timeZero(rtsdata)
        rtsdata = self.TimeCalibrationRTS(refclock,rtsdata,offset)
        return rtsdata

    def TimeCalibrationRTS(self,refclock,rtsclock,offset):
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
            rtsclock = self.timeZero(rtsclock)
        return rtsclock

    def timeZero(self,df):
        df['time'] = df['time']-df['time'][0]
        return df

    def reformat(self,times,df,t0):
        times = times.dropna()
        times = times.reset_index(drop=True)
        df = df.dropna()
        df = df.reset_index(drop=True)
        df = pd.concat([times,df],axis=1)
        if df['time'][0]<t0 and df['time'].iloc[-1]>t0:
            locs = len(df[(df['time'])<t0])
            df = df.drop(index=df.index[0:locs])
            df = df.reset_index(drop=True)
            df = self.timeZero(df)
            return df,t0
        elif df['time'][0]==t0:
            df = self.timeZero(df)   
            return df,t0 
        else:
            t0 = df['time'][0]
            df = self.timeZero(df)
            return df,t0
        
    def integrateS(self,df):
        displacements = np.zeros(df.shape)
        displacements = pd.DataFrame(displacements, columns=['time','s'])
        displacements.iloc[0,0] = df.iloc[0,0]
        displacements['time']= df['time']
        displacements['s'] = df.iloc[:,1].cumsum()
        return displacements

    def integrateV(self,df):
        displacements = self.copydfShape(df,0)
        displacements['time'] = df['time']
        displacements = displacements.rename(columns={'vel':'s'})
        dts = df['time'].diff()
        ds = df['vel']*dts
        displacements['s'] = ds.cumsum()
        displacements['s'][0] = 0    
        return displacements

    def s_to_ds(self,s):
        ds = self.copydfShape(s,0)
        ds['time'] = s['time']
        ds = ds.rename(columns={'s':'ds'})
        for i in range(1,len(s)):
            ds['ds'][i] = s['s'][i] - s['s'][i-1]
        return ds

    def s_to_v(self,s,span):
        vels = self.copydfShape(s,0)
        vels = vels.rename(columns={'s':'vel'})
        vels['time'] = s['time']
        dt = s['time'].diff()
        ds = s['s'].diff()
        vels['vel'] = ds/dt
        vels['vel'][0] = 0
        if span != 1:
            vels = self.smooth(vels,'vel',span)
        return vels

    def smooth(self,data,type,span):
        length = len(data)
        smoothData = self.copydfShape(data,0)
        smoothData['time'] = data['time']
        data = pd.concat([pd.DataFrame(np.zeros([int(m.ceil(span/2)),2]),columns=['time',type]),data],axis=0)
        data = pd.concat([data,pd.DataFrame(np.zeros([int(m.ceil(span/2)),2]),columns=['time',type])],axis=0)
        data = data.reset_index(drop=True)
        for i in range(m.ceil(span/2)-1,length - m.ceil(span/2) ):
            smoothData[type][i] = data[type][i-m.ceil(span/2):i+m.ceil(span/2)].mean()
        return smoothData
    
    def dropNan(self,df):
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def copydfShape(self,data,num):
        if num == 0:
            shapeCopy = np.zeros(data.shape)
        elif num == 1:
            shapeCopy = np.ones(data.shape)
        shapeCopy = pd.DataFrame(shapeCopy,columns=list(data.columns))
        return shapeCopy
    
    def frequencyCorrection(self,refPts, changePts):
        if len(changePts.columns) > 2:
            changePts = changePts.iloc[:,0:2]
        correction = self.copydfShape(refPts,0)
        correction.columns = changePts.columns
        for i in range(len(refPts)):
            refTime = refPts['time'][i]
            closest = changePts.iloc[(changePts['time']-refTime).abs().argsort()[:1]].astype('float')
            correction.iloc[i,:] = closest
        return correction

    def KvcerrorAnalysis(self,section):
        self.KvcErrPeak,self.KvcErrMean,self.KvcErrSTD = self.errorAnalysis(self.rts,self.Kvc,section)
        # return self.KvcErrPeak,self.KvcErrMean,self.KvcErrSTD
    
    def KwberrorAnalysis(self,section):
        self.KwbErrPeak,self.KwbErrMean,self.KwbErrSTD= self.errorAnalysis(self.rts,self.Kwb,section)
        return self.KwbErrPeak,self.KwbErrMean,self.KwbErrSTD

    def WBLerrorAnalysis(self,section):
        self.WBLErrPeak,self.WBLErrMean,self.WBLErrSTD = self.errorAnalysis(self.rts,self.s_wbl,section)
        return self.WBLErrPeak,self.WBLErrMean,self.WBLErrSTD

    def errorAnalysis(self,groundTruth,compData,section):
        compData = self.frequencyCorrection(groundTruth,compData)
        vcorr = self.frequencyCorrection(groundTruth,self.v_refL)
        up_section = vcorr[vcorr['vel'] >= 0]
        down_section = vcorr[vcorr['vel'] <= 0]
        up_section = up_section[up_section['time']<(up_section['time'].iloc[-1]*(2/3))]
        down_section = down_section[down_section['time']>(down_section['time'].iloc[-1]*(1/3))]
        # if section == 'down':
        #     initial_offset = groundTruth['s'].iloc[] - compData['s'].iloc[0]
        #     compData['s'] = compData['s'] + initial_offset
        
        compMax = compData['s'].max()
        gtMax = groundTruth['s'].max()
        peakErr = abs(compMax-gtMax)
        if section == 'up':
            compData_up = compData['s'].iloc[up_section.index]
            groundTruth_up = groundTruth['s'].iloc[up_section.index]
            compData_err = abs(compData_up - groundTruth_up)
            err_mean = compData_err.mean()
            err_std = compData_err.std()
        elif section == 'down':
            compData_down = compData['s'].iloc[down_section.index]
            groundTruth_down = groundTruth['s'].iloc[down_section.index]
            initial_offset = groundTruth_down.iloc[0] - compData_down.iloc[0]
            compData_down = compData_down + initial_offset
            compData_err = abs(compData_down - groundTruth_down)
            err_mean = compData_err.mean()
            err_std = compData_err.std()
        else:
            compData_err = abs(compData['s'] - groundTruth['s'])
            err_mean = compData_err.mean()
            err_std = compData_err.std()
        return peakErr,err_mean,err_std

    def applyTimeOffset(self,offsetTime,data):
        offset = self.copydfShape(data,0)
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
    
    def applyKVC(self,Q,Rm,Rwb,g_scale,span):
        self.KvcR = self.KVC(self.v_refR,self.ds_mrf,self.ds_mrb,self.ds_wbr,Q,Rm,Rwb,g_scale)
        self.KvcL = self.KVC(self.v_refL,self.ds_mlf,self.ds_mlb,self.ds_wbl,Q,Rm,Rwb,g_scale)
        self.KvcRVel = self.KvcR['vel']*self.KvcR['time'].diff()
        self.KvcLVel = self.KvcL['vel']*self.KvcL['time'].diff()
        self.KvcR['s'] = self.KvcRVel.cumsum()
        self.KvcL['s'] = self.KvcLVel.cumsum()
        self.KvcR['s'] = self.KvcR['s'] + self.absolute_height_offset
        self.KvcL['s'] = self.KvcL['s'] + self.absolute_height_offset
        self.Kvc = (self.KvcL+self.KvcR)/2
        if span != 1:
            self.KvcL['vel'] =  self.smooth(pd.concat([self.KvcL['time'],self.KvcL['vel']],axis=1),'vel',span)['vel']
            self.KvcR['vel'] =  self.smooth(pd.concat([self.KvcR['time'],self.KvcR['vel']],axis=1),'vel',span)['vel']  
            self.Kvc['vel'] =  self.smooth(pd.concat([self.Kvc['time'],self.Kvc['vel']],axis=1),'vel',span)['vel']
        if self.Kvc['s'][0].any():
            self.Kvc['s'][0] = self.Kvc['s'][1]
        self.Kvc.columns = ['time','s','vel']
        
    def KVC_to_UTScope_data(self):
        s = self.Kvc['s']*3.28084
        time = self.Kvc['time']
        self.UTScopeData = pd.concat([time,s],axis=1)
        self.UTScopeData.columns = ['time','Total s']

    def applyKWB(self,Q,Rwb,span):
        self.Kwb = self.KWB(self.v_refR,self.v_refL,self.ds_wbr,self.ds_wbl,Q,Rwb)
        if span != 1:
            self.Kwb['vel'] =  self.smooth(pd.concat([self.Kwb['time'],self.Kwb['vel']],axis=1),'vel',span)['vel']
        self.Kwb.columns = ['time','s','vel']


    def KVC(self,v_ref,ds_f,ds_c,ds_b,Q,Rm,Rwb,g_scale):
            '''
            Final Kalman Filter that compensates for velocity errors in all encoders and fuses them together. Meant to be used for one
            side of the robot, filtering 2 motor encoders and 1 wheelie bar encoder'''

            if self.units == 'm':
                motor_velocity_threshold = 0.01
                motor_difference_threshold = 0.01
                wheeliebar_threshold_ratio = 0.10668

            elif self.units == 'ft':
                motor_velocity_threshold = 0.0328084
                motor_difference_threshold = 0.0328084
                wheeliebar_threshold_ratio = 0.25

            kalman_data_frame = np.zeros([len(ds_f),3])
            kalman_data_frame = pd.DataFrame(kalman_data_frame, columns=['time','s','vel'])
            start_id = 4
            ds_start_avg = (ds_f.iloc[0:start_id-1,:] + ds_c.iloc[0:start_id-1,:] + ds_b.iloc[0:start_id-1,:])/3
            kalman_data_frame.iloc[0:start_id-1,0:2] = ds_start_avg
            kalman_data_frame.iloc[0:start_id-1,2] = ds_start_avg['ds']/ds_start_avg['time'].diff()
            dv = v_ref['vel'].diff()
            #x-matrix = [s v verr]
            time = ds_f['time']
            dt = time.diff()
            v_ref = v_ref['vel']
            #state = [t s v verr]
            kalman_data_frame.iloc[:,0] = time
            kalman_data_frame.iloc[0,1:3] = kalman_data_frame.iloc[1,1:3]
            Pprev = Q
            Xprev = np.matrix([[0],[v_ref[0]]]).astype('float64')

            for i in range(start_id,len(ds_f)):
                dt_i = dt[i]
                dv_i = dv[i]
                F = np.matrix([[1,dt_i],
                            [0,1]])
                H = np.matrix([[1,dt_i],
                            [0,1]])

                curr_vref = v_ref[i]
                vprev = Xprev[1]
                v = vprev+dv_i

                if curr_vref < 0:
                    v = v * g_scale      
                Xprev[1] = v

                #prediction step
                Xpred = F*Xprev
                Ppred = F*Pprev*F.T+Q

                v_f = ds_f.iloc[i,1]/dt_i
                # set z (observer state) based on encoder values
                Xobs_f = np.matrix([[Xprev[0,0] + ds_f.iloc[i,1]],
                                    [v_f]]).astype('float64')
                v_c = ds_c.iloc[i,1]/dt_i
                Xobs_c = np.matrix([[Xprev[0,0] + ds_c.iloc[i,1]],
                                    [v_c]]).astype('float64')
                v_m = (v_c + v_f)/2
                Xobs_m = (Xobs_c + Xobs_f)/2

                if Xobs_m.max() > 100:
                    return kalman_data_frame
                v_b = ds_b.iloc[i,1]/dt_i
                Xobs_b = np.matrix([[Xprev[0,0] + ds_b.iloc[i,1]],
                                    [v_b]]).astype('float64')
                
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
                wb_avg = float(0.2*Xobs_b[1]+0.2*vwb_prev1+0.2*vwb_prev2+0.2*vwb_prev3+0.2*vwb_prev4)
                mb_avg = float(0.2*Xobs_c[1]+0.2*vmb_prev1+0.2*vmb_prev2+0.2*vmb_prev3+0.2*vmb_prev4)
                mf_avg = float(0.2*Xobs_f[1]+0.2*vmf_prev1+0.2*vmf_prev2+0.2*vmf_prev3+0.2*vmf_prev4)
                wb_diff = abs(curr_vref - wb_avg)
                mb_diff = abs(curr_vref - mb_avg)
                mf_diff = abs(curr_vref - mf_avg)
                m_diff = abs(mb_avg-mf_avg)
                # goes throught different cases
                # motors cannot slip if wheelie bar is slipping
                # if wb_diff > abs(curr_vref*wheeliebar_threshold_ratio) or abs(v_b) > wheeliebar_velocity_threshold:
                if wb_diff > abs(curr_vref*wheeliebar_threshold_ratio):
                    if (mf_diff > motor_velocity_threshold and mb_diff > motor_velocity_threshold and curr_vref != 0) or (m_diff > motor_difference_threshold):
                        if m_diff > 6*motor_difference_threshold:
                            # print(time[i],m_diff,'wb & motor diff')
                            gainM = np.zeros((2,2)) 
                            gainWB = Ppred*H.T*inv(H*Ppred*H.T + 2*Rwb)
                        elif (abs(v_b) == 0 and curr_vref != 0):
                            # print(time[i],wb_diff,'wb inaccuracy')
                            gainM = Ppred*H.T*inv(H*Ppred*H.T + Rm)
                            gainWB = np.zeros((2,2))   
                        else:
                            # print(time[i],wb_diff,'wb & motor slip')
                            gainM = Ppred*H.T*inv(H*Ppred*H.T + 2*Rm)
                            gainWB = Ppred*H.T*inv(H*Ppred*H.T + 2*Rwb)
                        motor_adj = gainM*(Xobs_m-Xpred)
                    else:
                        # print(time[i],wb_diff,'wb slip')
                        gainM = Ppred*H.T*inv(H*Ppred*H.T + Rm)
                        gainWB = np.zeros((2,2))   
                        motor_adj = gainM*(Xobs_m-Xpred)

                elif m_diff > motor_difference_threshold:
                    # print(time[i],m_diff,'Motor Diff')
                    motor_adj = 0

                elif mf_diff > motor_velocity_threshold and mb_diff < motor_velocity_threshold and curr_vref != 0:
                    # print(time[i],mf_diff,'Front Motor slip')
                    motor_adj = gainM*(Xobs_c-Xpred)

                elif mf_diff < motor_velocity_threshold and mb_diff > motor_velocity_threshold and curr_vref != 0:
                    # print(time[i],mf_diff,'Back Motor slip')
                    motor_adj = gainM*(Xobs_f-Xpred)
                
                #this condition takes away emphasis of motors
                elif mf_diff > motor_velocity_threshold and mb_diff > motor_velocity_threshold and curr_vref != 0:
                    # print(time[i],mf_diff,'ALL Motor slip')
                    motor_adj = 0                   

                else:
                    gainM = Ppred*H.T*inv(H*Ppred*H.T + 1.5*Rm)
                    motor_adj = gainM*(Xobs_m-Xpred)
                    gainWB = Ppred*H.T*inv(H*Ppred*H.T + 1.5*Rwb)
                    
                    
                #multiply gain by difference between measured and predicted
                Xpred = Xpred + motor_adj + gainWB*(Xobs_b-Xpred)
                Ppred = Ppred - gainM*H*Ppred - gainWB*H*Ppred

                # if ((Xpred - Xprev)[0] < -0.01 and curr_vref > 0) or ((Xpred - Xprev)[0] > 0.01 and curr_vref < 0):
                #     print(time[i],(Xpred - Xprev)[0],'negative movement')

                # if RC cuts out and no data is logged
                if dt_i > 0.5:
                    Xpred[0] = (Xobs_m[0]+Xobs_b[0])/2
                    Xpred[1] = (v_m + v_b)/2
                    Ppred = Q
                    # print(time[i],'time trigger')

                #increment current to past
                Xprev = Xpred
                Pprev = Ppred

                kalman_data_frame.iloc[i,1:3] = Xpred.T
            return kalman_data_frame

    def KWB(self,v_refR,v_refL,ds_wbr,ds_wbl,Q,Rwb):
        '''Kalman filter for filtering between left and right wheelie bar '''
        thresh = 0.1
        kalman_data_frame = np.zeros([len(ds_wbl),3])
        kalman_data_frame = pd.DataFrame(kalman_data_frame, columns=['time','s','vel'])
        dvL = v_refL['vel'].diff()
        dvR = v_refR['vel'].diff()

        time = ds_wbl['time']
        dt = time.diff()
        v_refL = v_refL['vel']
        v_refR = v_refR['vel']

        #state = [t s v verr]
        kalman_data_frame.iloc[:,0] = time

        Pprev = Q
        Xprev = np.matrix([[0],[0]]).astype('float64')
        for i in range(4,len(ds_wbl)):
            F = np.matrix([[1,dt[i]],
                        [0,1]])
            H = np.matrix([[1,dt[i]],
                        [0,1]])
            vprev = Xprev[1]
            v = vprev+(dvR[i]+dvL[i])/2    
            Xprev[1] = v

            #prediction step
            Xpred = F*Xprev
            Ppred = F*Pprev*F.T+Q

            # set z (observer state) based on encoder values
            Xobs_l = np.matrix([[Xprev[0,0] + ds_wbl.iloc[i,1]],
                                [ds_wbl.iloc[i,1]/dt[i]]]).astype('float64')
            Xobs_r = np.matrix([[Xprev[0,0] + ds_wbr.iloc[i,1]],
                                [ds_wbr.iloc[i,1]/dt[i]]]).astype('float64')
            
            if Xobs_l.max() > 100 or Xobs_r.max() > 100:
                return kalman_data_frame
            
            vl_prev1 = ds_wbl.iloc[i-1,1]/dt[i-1]
            vl_prev2 = ds_wbl.iloc[i-2,1]/dt[i-2]
            vl_prev3 = ds_wbl.iloc[i-3,1]/dt[i-3]
            vl_prev4 = ds_wbl.iloc[i-4,1]/dt[i-4]
            vr_prev1 = ds_wbr.iloc[i-1,1]/dt[i-1]
            vr_prev2 = ds_wbr.iloc[i-2,1]/dt[i-2]
            vr_prev3 = ds_wbr.iloc[i-3,1]/dt[i-3]
            vr_prev4 = ds_wbr.iloc[i-4,1]/dt[i-4]
            

            wbl_avg = 0.2*Xobs_l[1]+0.2*vl_prev1+0.2*vl_prev2+0.2*vl_prev3+0.2*vl_prev4
            wbl_diff = abs(v_refL[i] - wbl_avg)
            wbr_avg = 0.2*Xobs_r[1]+0.2*vr_prev1+0.2*vr_prev2+0.2*vr_prev3+0.2*vr_prev4
            wbr_diff = abs(v_refR[i] - wbr_avg)
            #when left WB slips
            if wbl_diff > thresh and wbr_diff < thresh:
                # print(time[i],"left slip")
                gainR = Ppred*H.T*inv(H*Ppred*H.T + Rwb)
                gainL = np.zeros((2,2))

            #when right WB slips
            elif wbr_diff > thresh and wbl_diff < thresh:
                # print(time[i],"right slip")
                gainR = np.zeros((2,2))
                gainL = Ppred*H.T*inv(H*Ppred*H.T + Rwb)
            #measurement step
            else:
                gainL = Ppred*H.T*inv(H*Ppred*H.T + Rwb)
                gainR = Ppred*H.T*inv(H*Ppred*H.T + Rwb)

            #multiply gain by difference between measured and predicted
            Xpred = Xpred + gainR*(Xobs_r-Xpred) + gainL*(Xobs_l-Xpred)
            Ppred = Ppred - gainR*H*Ppred - gainL*H*Ppred

            #increment current to past
            Xprev = Xpred
            Pprev = Ppred

            kalman_data_frame.iloc[i,1:3] = Xpred.T
        return kalman_data_frame

    def run_Kall_analysis(self,Q,Rm,Rwb,g_scale,section):
        try:
            self.applyKVC(Q,Rm,Rwb,g_scale,1)
            self.KvcerrorAnalysis(section)
            self.WBLerrorAnalysis(section)
        except Exception as e:
            return e
    
    def plot_left_distance(self,stopPoint):
        if self.rts is not None:
            mpl.plot(self.rts.loc[self.rts['time'] < stopPoint]['time'],self.rts.loc[self.rts['time'] < stopPoint]['s'],label='RTS')
        mpl.plot(self.s_wbl.loc[self.s_wbl['time'] < stopPoint]['time'],self.s_wbl.loc[self.s_wbl['time'] < stopPoint]['s'],label='Left Wheelie Bar')
        mpl.plot(self.s_mlb.loc[self.s_mlb['time'] < stopPoint]['time'],self.s_mlb.loc[self.s_mlb['time'] < stopPoint]['s'],label='Left Back Motor')
        mpl.plot(self.s_mlf.loc[self.s_mlf['time'] < stopPoint]['time'],self.s_mlf.loc[self.s_mlf['time'] < stopPoint]['s'],label='Left Front Motor')
        if self.KvcL is not None:
            mpl.plot(self.KvcL['time'],self.KvcL['s'],label='Left KVC')
        mpl.legend()
        mpl.title('Left: Height vs Time')
        mpl.xlabel('Time (s)')
        if self.units == 'm':
            mpl.ylabel('Height (m)')
        elif self.units == 'ft':
            mpl.ylabel('Height (ft)')
    
    def plot_right_distance(self,stopPoint):
        if self.rts is not None:
            mpl.plot(self.rts.loc[self.rts['time'] < stopPoint]['time'],self.rts.loc[self.rts['time'] < stopPoint]['s'],label='RTS')
        mpl.plot(self.s_wbr.loc[self.s_wbr['time'] < stopPoint]['time'],self.s_wbr.loc[self.s_wbr['time'] < stopPoint]['s'],label='Right Wheelie Bar')
        mpl.plot(self.s_mrb.loc[self.s_mrb['time'] < stopPoint]['time'],self.s_mrb.loc[self.s_mrb['time'] < stopPoint]['s'],label='Right Back Motor')
        mpl.plot(self.s_mrf.loc[self.s_mrf['time'] < stopPoint]['time'],self.s_mrf.loc[self.s_mrf['time'] < stopPoint]['s'],label='Right Front Motor')
        if self.KvcR is not None:
            mpl.plot(self.KvcR['time'],self.KvcR['s'],label='Right KVC')
        mpl.legend()
        mpl.title('Right: Height vs Time')
        mpl.xlabel('Time (s)')
        if self.units == 'm':
            mpl.ylabel('Height (m)')
        elif self.units == 'ft':
            mpl.ylabel('Height (ft)')

    def plot_right_velocity(self):
        mpl.plot(self.v_wbr['time'],self.v_wbr['vel'],linewidth=0.5,label='Right Wheelie Bar')
        mpl.plot(self.v_mrb['time'],self.v_mrb['vel'],linewidth=0.5,label='Right Back Motor')
        mpl.plot(self.v_mrf['time'],self.v_mrf['vel'],linewidth=0.5,label='Right Front Motor')
        if self.KvcR is not None:
            mpl.plot(self.KvcR['time'],self.KvcR['vel'],linewidth=0.5,label='Right KVC')
        mpl.plot(self.v_refR['time'],self.v_refR['vel'],linewidth=1.0,label='Reference Velocity')
        mpl.title('Right: Velocity vs Time')
        mpl.legend()
        mpl.xlabel('Time (s)')
        if self.units == 'm':
            mpl.ylabel('Velocity (m/s)')
        elif self.units == 'ft':
            mpl.ylabel('Velocity (ft/s)')
    
    def plot_left_velocity(self):
        mpl.plot(self.v_wbl['time'],self.v_wbl['vel'],linewidth=0.5,label='Left Wheelie Bar')
        mpl.plot(self.v_mlb['time'],self.v_mlb['vel'],linewidth=0.5,label='Left Back Motor')
        mpl.plot(self.v_mlf['time'],self.v_mlf['vel'],linewidth=0.5,label='Left Front Motor')
        if self.KvcL is not None:
            mpl.plot(self.KvcL['time'],self.KvcL['vel'],linewidth=0.5,label='Left KVC')
        mpl.plot(self.v_refL['time'],self.v_refL['vel'],linewidth=1.0,label='Reference Velocity')
        mpl.title('Left: Velocity vs Time')
        mpl.legend()
        mpl.xlabel('Time (s)')
        if self.units == 'm':
            mpl.ylabel('Velocity (m/s)')
        elif self.units == 'ft':
            mpl.ylabel('Velocity (ft/s)')
    
    def plot_total_distance(self):
        if self.rts is not None:
            mpl.plot(self.rts['time'],self.rts['s'],label='RTS')
        if self.Kvc is not None:
            mpl.plot(self.Kvc['time'],self.Kvc['s'],label='KVC')
        # if self.RC_position is not None:
        #     mpl.plot(self.RC_position['time'],self.RC_position['s'],label='RC Position')
        mpl.plot(self.s_wbr['time'],self.s_wbr['s'],label='Right Wheelie Bar')
        mpl.plot(self.s_wbl['time'],self.s_wbl['s'],label='Left Wheelie Bar')
        mpl.title('Height vs Time')
        mpl.legend()
        mpl.xlabel('Time (s)')
        if self.units == 'm':
            mpl.ylabel('Height (m)')
        elif self.units == 'ft':
            mpl.ylabel('Height (ft)')

    def plot_downward(self):
        down_section = self.v_refL[self.v_refL['vel'] <= 0]
        down_section = down_section['time'][down_section['time']>(down_section['time'].iloc[-1]*(1/3))]
        Kvc_down = self.Kvc[self.Kvc['time'] > down_section.iloc[0]]
        RTS_down = self.rts[self.rts['time'] > down_section.iloc[0]]
        initial_offset = RTS_down['s'].iloc[0] - Kvc_down['s'].iloc[0]
        Kvc_down['s'] = Kvc_down['s'] + initial_offset
        mpl.plot(RTS_down['time'],RTS_down['s'],label='RTS')
        mpl.plot(Kvc_down['time'],Kvc_down['s'],label='KVC')

        mpl.title('Height vs Time')
        mpl.legend()
        mpl.xlabel('Time (s)')
        if self.units == 'm':
            mpl.ylabel('Height (m)')
        elif self.units == 'ft':
            mpl.ylabel('Height (ft)')

    def save_total_distance_plot(self,path):
        if self.rts is not None:
            mpl.plot(self.rts['time'],self.rts['s'],label='RTS')
        if self.Kvc is not None:
            mpl.plot(self.Kvc['time'],self.Kvc['s'],label='KVC')
        mpl.plot(self.s_wbr['time'],self.s_wbr['s'],label='Right Wheelie Bar')
        mpl.plot(self.s_wbl['time'],self.s_wbl['s'],label='Left Wheelie Bar')
        mpl.title('Height vs Time')
        mpl.legend()
        mpl.xlabel('Time (s)')
        if self.units == 'm':
            mpl.ylabel('Height (m)')
        elif self.units == 'ft':
            mpl.ylabel('Height (ft)')
        mpl.savefig(path)
        mpl.close()