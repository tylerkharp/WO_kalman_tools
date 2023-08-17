import numpy as np
import matplotlib.pyplot as mpl
import ObjDataManipulation as odm
import calibration_tools as ct
import time

saveKalman = False
ref_offset = 0.20
GT_offset = .30
# g_scale = 1.017733333
# g_scale = 1.022178

# g_scale = 1.0
section = 'down'
span = 1

prevTime = time.time()
# runData = odm.runData('/home/tyler.harp/dev/ekkocortex/data/CSVData/cedarData/run64.csv',0.20,0.30,'Cedar','m',False)
# runData = odm.runData('/home/tyler.harp/dev/ekkocortex/data/CSVData/montiData/monticello_18_44_07.csv',0.20,0.30,'Monti','m',0)
# runData = odm.runData('/home/tyler.harp/dev/ekkocortex/data/CSVData/TerrariumData/20230427-e92647/runData/G4-0040_run_0001.csv',
#                       0.20,0,'Terrarium','ft')
# runData = odm.runData('/home/tyler.harp/dev/ekkocortex/data/CSVData/TerrariumData/20230515-608966/runData/G4-0093_run_0080.csv',
#                       0.20,0,'Terrarium','ft')
# runData = odm.runData('/home/tyler.harp/UTScopeData/20230427-e92647/raw_run_data/G4-0040_run_0107.csv',
#                       0.20,0,'Terrarium','ft',0)
# runData = odm.runData('/home/tyler.harp/UTScopeData/20230516-3c4686/raw_run_data/G4-0001_run_0136.csv',
#                       0.20,0,'Terrarium','ft',0)
# runData = odm.runData('/home/tyler.harp/UTScopeData/20230522-1ac0e6/raw_run_data/G4-0080_run_0060.csv',
#                       0.20,0,'Terrarium','ft',0)
runData = odm.runData('/home/tyler.harp/UTScopeData/20230131-7e2700/raw_run_data/G4-0047_run_0019.csv',
                      0.20,0,'Terrarium','ft',0)
prevTime = ct.get_tDiff(prevTime)

runData.getVelocities(span)
prevTime = ct.get_tDiff(prevTime)

#before terrarium
# Q_set = 0.6311111111
# Rm_set = 0.4677777778
# Rwb_set = 0.440555555555556

#after terrarium
# Q_set = 0.712778
# Rm_set = 0.277222
# Rwb_set = 0.25
# g_scale = 1.0
#with Validation
Q_set = 0.8488891111
Rm_set = 0.522222
Rwb_set = 0.4405555556
g_scale = 1.022178


Q = np.identity(2)*Q_set
Rm = np.identity(2)*Rm_set
Rwb = np.identity(2)*Rwb_set

# runData.run_Kall_analysis(Q,Rm,Rwb,QKwb,RKw20230508-04487eb,g_scale,section)
runData.applyKVC(Q,Rm,Rwb,g_scale,span)
# runData.applyKWB(QKwb,RKwb)
# runData.KvcerrorAnalysis(section)
# runData.WBLerrorAnalysis(section)
# runData.KwberrorAnalysis(section)
# print(runData.KvcErrMean,runData.KvcErrPeak,runData.WBLErrMean,runData.WBLErrPeak)

f1 = mpl.figure()
runData.plot_left_distance(1000000)
f2 = mpl.figure()
runData.plot_right_distance(1000000)
f3 = mpl.figure()
runData.plot_left_velocity()
f4 = mpl.figure()
runData.plot_right_velocity()
f5 = mpl.figure()
runData.plot_total_distance()
# f6 = mpl.figure()
# runData.plot_downward()

mpl.show()
