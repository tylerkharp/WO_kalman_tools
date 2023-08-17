import pandas as pd
import numpy as np
import time 
import ObjDataManipulation as odm
import multiprocessing as mp


def get_tDiff(prevTime):
    print(time.time()-prevTime)
    prevTime = time.time()
    return prevTime

def initiate_runData(param_list):
    try:
        data = odm.runData(param_list[0],param_list[1],param_list[2],param_list[3],param_list[4],param_list[5])
        return data
    except Exception as e:
        return e
    
def initiate_data_list(param_list,numCPU):
    data_list = np.zeros(len(param_list),dtype=object)
    procs = mp.Pool(numCPU).map(initiate_runData,param_list)
    count = 0
    for results in procs:
        data_list[count] = results
        count += 1
    return data_list


def run_and_analyse_AKF(k_list):
    try:
        k_list[0].applyAKF(k_list[1],k_list[2],k_list[3],k_list[4])
        k_list[0].KFerrorAnalysis('whole')
        return k_list[0].KFerrMean
    except Exception as e:
        return e
    
def run_and_analyse_KF(k_list):
    try:
        k_list[0].applyKF(k_list[1],k_list[2],k_list[3],k_list[4])
        k_list[0].KFerrorAnalysis('whole')
        return k_list[0].KFerrMean
    except Exception as e:
        return e

def run_and_analyse_Kvc(k_list):
    try: 
        k_list[0].applyKVC(k_list[1],k_list[2],k_list[3],k_list[4],1)
        k_list[0].KvcerrorAnalysis(k_list[5])
        return [k_list[0].KvcErrMean,k_list[0].KvcErrPeak,k_list[0].KvcErrSTD]
    except Exception as e:
        return e

def run_and_analyse_all(param_list):
    data = param_list[0]
    Q = param_list[1]
    Rm = param_list[2]
    Rwb = param_list[3]
    gscale = param_list[4]
    section = param_list[5]
    data.run_Kall_analysis(Q,Rm,Rwb,gscale,section)
    return data.KvcErrMean


def gscale_sweep(param_list):
    data = param_list[0]
    Q = param_list[1]
    Rm = param_list[2]
    Rwb = param_list[3]
    gscale_range = param_list[4]
    success_matrix = np.zeros([len(gscale_range),4])
    success_matrix = pd.DataFrame(success_matrix,columns=['Gscale','errMean','errPeak','errSTD'])
    for i in range(len(gscale_range)):
        errMean,errPeak,errSTD = run_and_analyse_Kvc([data,Q,Rm,Rwb,gscale_range[i],'down'])
        success_matrix.iloc[i,:] = [gscale_range[i],errMean,errPeak,errSTD]
    return success_matrix

def run_gscale_sweep(data_list,Q,Rm,Rwb,gscale_range):
    Q = np.identity(2)*Q
    Rm = np.identity(2)*Rm
    Rwb = np.identity(2)*Rwb
    param_list = np.zeros([len(data_list),5],dtype=object)
    for i in range(len(data_list)):
        param_list[i,:] = [data_list[i],Q,Rm,Rwb,gscale_range]
    procs = mp.Pool(12).map(gscale_sweep,param_list)
    sumErr = 0
    for results in procs:
        sumErr += results.iloc[:,1]
    meanErr = sumErr/len(param_list)
    success_matrix = pd.concat([procs[0].iloc[:,0],meanErr],axis=1)
    return success_matrix

def R_sweep(param_list):
    data_list = param_list[0]
    Q = np.identity(2)*param_list[1]
    Rm_range = param_list[2]
    Rwb_range = param_list[3]
    gscale = param_list[4]
    success_matrix = np.zeros([len(Rm_range)*len(Rwb_range),7])
    success_matrix = pd.DataFrame(success_matrix,columns=['Q','Rm','Rwb','KFErr','KvcErr','KwbErr','WBLErr'])
    index = 0
    for i in range(len(Rm_range)):
        prevTime = time.time()
        for j in range(len(Rwb_range)):
            Rm = np.identity(2)*Rm_range[i]
            Rwb = np.identity(2)*Rwb_range[j]
            KFsumErr = 0
            KVCsumErr = 0
            KWBsumErr = 0
            WBLsumErr = 0
            for data in data_list:
                e_KF,e_Kvc,e_Kwb,e_WBL = run_and_analyse_all(data,Q,Rm,Rwb,gscale,'up')
                KFsumErr += e_KF
                KVCsumErr += e_Kvc
                KWBsumErr += e_Kwb
                WBLsumErr += e_WBL
            KFmeanErr = KFsumErr/len(data_list)
            KVCmeanErr = KVCsumErr/len(data_list)
            KWBmeanErr = KWBsumErr/len(data_list)
            WBLmeanErr = WBLsumErr/len(data_list)
            success_matrix.iloc[index,:] = [param_list[1],Rm_range[i],Rwb_range[j],KFmeanErr,KVCmeanErr,KWBmeanErr,WBLmeanErr]
            index += 1
            prevTime = get_tDiff(prevTime)
    return success_matrix

def run_QR_sweep(data_list,Q_range,Rm_range,Rwb_range):
    param_list = np.zeros([len(Q_range),5],dtype=object)
    for i in range(len(Q_range)):
        param_list[i,:] = [data_list,Q_range[i],Rm_range,Rwb_range,1.0]
        # test = R_sweep(param_list[i,:])
    procs = mp.Pool(10).map(R_sweep,param_list)
    success_matrix = 0
    for results in procs:
        if results == procs[0]:
            success_matrix = results
        else:
            success_matrix = pd.concat([success_matrix,results],axis=0)
    return success_matrix

def run_QR_sweep_isolated(data_list,Q_range,Rm_range,Rwb_range,gscale):
    # prevtime = time.time()
    success_matrix = np.zeros([len(Q_range)*len(Rm_range)*len(Rwb_range),4])
    success_matrix = pd.DataFrame(success_matrix,columns=['Q','Rm','Rwb','KvcErr'])
    index = 0
    k_list = np.zeros((len(data_list),6),dtype=object)
    for i in range(len(Q_range)):
        for j in range(len(Rm_range)):
            for k in range(len(Rwb_range)):
                Q = np.identity(2)*Q_range[i]
                Rm = np.identity(2)*Rm_range[j]
                Rwb = np.identity(2)*Rwb_range[k]
                for ind in range(len(data_list)):
                    k_list[ind,:] = [data_list[ind],Q,Rm,Rwb,gscale,'up']
                procs = mp.Pool(12).map(run_and_analyse_all,k_list)
                KVCsumErr = 0
                for results in procs:
                    KVCsumErr += results
                KVCmeanErr = KVCsumErr/len(data_list)
                success_matrix.iloc[index,:] = [Q_range[i],Rm_range[j],Rwb_range[k],KVCmeanErr]
                index += 1
    return success_matrix

def run_QR_KVC_sweep_isolated(data_list,Q_range,Rm_range,Rwb_range,gscale):
    # prevtime = time.time()
    success_matrix = np.zeros([len(Q_range)*len(Rm_range)*len(Rwb_range),5])
    success_matrix = pd.DataFrame(success_matrix,columns=['Q','Rm','Rwb','KvcErr'])
    index = 0
    k_list = np.zeros((len(data_list),6),dtype=object)
    for i in range(len(Q_range)):
        for j in range(len(Rm_range)):
            for k in range(len(Rwb_range)):
                Q = np.identity(2)*Q_range[i]
                Rm = np.identity(2)*Rm_range[j]
                Rwb = np.identity(2)*Rwb_range[k]
                for ind in range(len(data_list)):
                    k_list[ind,:] = [data_list[ind],Q,Rm,Rwb,gscale,'up']
                procs = mp.Pool(12).map(run_and_analyse_all,k_list)
                KVCsumErr = 0
                KWBsumErr = 0
                for results in procs:
                    KVCsumErr += results[0]
                    KWBsumErr += results[1]
                KVCmeanErr = KVCsumErr/len(data_list)
                KWBmeanErr = KWBsumErr/len(data_list)
                success_matrix.iloc[index,:] = [Q_range[i],Rm_range[j],Rwb_range[k],KVCmeanErr,KWBmeanErr]
                index += 1
    return success_matrix

def error_analysis_sweep(data_list,Q,Rm,Rwb,gscale,section):
    WBLsumErr = 0
    KVCsumErr = 0
    Q = np.identity(2)*Q
    Rm = np.identity(2)*Rm
    Rwb = np.identity(2)*Rwb
    error_matrix = np.zeros((len(data_list),4))
    error_matrix = pd.DataFrame(error_matrix,columns=['KvcErr','KvcPeakErr','WBLErr','WBLPeakErr'])

    for i in range(len(data_list)):
        data_list[i].run_Kall_analysis(Q,Rm,Rwb,gscale,section)
        error_matrix.iloc[i,:] = [data_list[i].KvcErrMean,data_list[i].KvcErrPeak,data_list[i].WBLErrMean,data_list[i].WBLErrPeak]
        KVCsumErr += data_list[i].KvcErrMean
        WBLsumErr += data_list[i].WBLErrMean
    WBLmeanErr = WBLsumErr/len(data_list)
    KVCmeanErr = KVCsumErr/len(data_list)
    return KVCmeanErr,WBLmeanErr,error_matrix




        

        