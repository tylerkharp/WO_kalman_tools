import pandas as pd
import numpy as np
import ObjDataManipulation as odm
import time
import calibration_tools as ct
import multiprocessing as mp

prevTime = time.time()
ref_offset = 0.20
GT_offset = .3017733333

span = 10 #10
width = 0.04 #0.49
#after terrarium
# 0.8488891111	0.522222	0.4405555556
# Q_set = 0.712778
# Rm_set = 0.277222
# Rwb_set = 0.25

Q_set = 0.8488891111
Rm_set = 0.522222
Rwb_set = 0.4405555556
# g_scale_set = 1.180556
g_scale_set = 1.031066


Q_range = np.linspace(Q_set-width/2,Q_set+width/2,num=span)
Rm_range = np.linspace(Rm_set-width/2,Rm_set+width/2,num=span)
Rwb_range = np.linspace(Rwb_set-width/2,Rwb_set+width/2,num=span)
gscale_range = np.linspace(g_scale_set-width/2,g_scale_set+width/2,num=span)
Q = np.identity(2)*Q_set
Rm = np.identity(2)*Rm_set
Rwb = np.identity(2)*Rwb_set

param_list = [['/home/tyler.harp/dev/ekkocortex/data/CSVData/cedarData/run58.csv',ref_offset,GT_offset,'Cedar','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/cedarData/run62.csv',ref_offset,GT_offset,'Cedar','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/cedarData/run70.csv',ref_offset,GT_offset,'Cedar','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/cedarData/run74.csv',ref_offset,0,'Cedar','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/montiData/monticello_20_51_03.csv',ref_offset,GT_offset,'Monti','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/montiData/monticello_19_31_00.csv',ref_offset,GT_offset,'Monti','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/montiData/monticello_18_44_07.csv',ref_offset,GT_offset,'Monti','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/montiData/monticello_18_39_40.csv',ref_offset,GT_offset,'Monti','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/cedarData/run60.csv',ref_offset,GT_offset,'Cedar','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/cedarData/run64.csv',ref_offset,GT_offset,'Cedar','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/cedarData/run66.csv',ref_offset,GT_offset,'Cedar','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/cedarData/run68.csv',ref_offset,GT_offset,'Cedar','m',0],
            ['/home/tyler.harp/dev/ekkocortex/data/CSVData/cedarData/run72.csv',ref_offset,1.5,'Cedar','m',0],]
if __name__ == "__main__":
    data_list = ct.initiate_data_list(param_list,6)
    prevTime = ct.get_tDiff(prevTime)

KVCmeanErr,WBLmeanErr,error_matrix = ct.error_analysis_sweep(data_list,Q,Rm,Rwb,g_scale_set,'up')
error_matrix.to_excel(r'/home/tyler.harp/dev/ekkocortex/data/kalmanData/success_matrix_withAllTerrarium.xlsx',engine='openpyxl')

print(KVCmeanErr,WBLmeanErr)
print(error_matrix)
prevTime = ct.get_tDiff(prevTime)

# if __name__ == "__main__":
#     # success_matrix = ct.run_QR_sweep_isolated(data_list,Q_range,Rm_range,Rwb_range,g_scale_set)
#     success_matrix = ct.run_gscale_sweep(data_list,Q_set,Rm_set,Rwb_set,gscale_range)
#     prevTime = ct.get_tDiff(prevTime)
#     success_matrix.to_excel(r'/home/tyler.harp/dev/ekkocortex/data/kalmanData/success_matrix_gscalewithValidation.xlsx',engine='openpyxl')
# print(success_matrix)






