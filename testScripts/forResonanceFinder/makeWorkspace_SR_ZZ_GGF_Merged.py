# an ExampleAnalysis driver for the running an example script
import ROOT
from ROOT import RF as RF

import numpy as np
import array
import sys

def findBinning(region,usePNN):
  bins=[]
  if not usePNN:
    if "Res" in region:
      if "L2" in region: bins = [300,320,350,380,410,440,470,500,530,560,600,640,680,720,770,820,870,930,990,1060,1130,1210,1290,1380,1470,1570,1680,1790,2140,3000]
      if "L1" in region: bins = [300,340,390,440,490,540,590,650,710,770,840,910,990,1070,1160,1250,1350,1460,1570,1690,1820,1970,2120,3000]
    elif "Merg" in region:
      print("In Merg")
      if "L2" in region:
        print("Merged regime; two-leptons")
        bins = [500,530,570,610,650,690,730,770,810,850,890,930,970,1010,1050,1100,1150,1200,1250,1300,1350,1410,1470,1530,1590,1650,1720,1790,1860,1930,2000,2080,2160,2240,2330,2510,2690,2870,3320,3770,6000]
      if "L1" in region:
        bins = [500,560,630,700,770,840,910,980,1050,1120,1200,1280,1360,1440,1530,1620,1710,1800,1900,2000,2100,2200,2310,2420,2530,2650,2770,2890,3210,3530,4320,6000]
      if "L0" in region:
        bins = [500,540,590,640,690,740,790,840,900,960,1020,1080,1150,1220,1290,1370,1450,1530,1620,1710,1800,1900,2010,2120,2230,2350,2470,2680,2890,3100,3310,6000]
    '''
    if doSingleBinCR and "CR" in region:
      if "Res" in region: bins=[300,3000]
      if "Merg" in region: bins=[500,6000]
    '''
  if usePNN:
    bins=np.linspace(0,1,200)
    #bins=np.linspace(0,1,1000)

  return bins

def findDSID(region, signal, mass):
  dsidmap={}
  if "L0" in region:
    dsidmap={"HVTWZ_400":307369, "HVTWZ_500":302241, "HVTWZ_600":302242, "HVTWZ_700":302243, "HVTWZ_800":302244, "HVTWZ_1000":302246, "HVTWZ_1200":302248, "HVTWZ_1400":302250, "HVTWZ_1500":302251, "HVTWZ_1600":302252, "HVTWZ_1800":302254, "HVTWZ_2000":302256, "HVTWZ_2400":302258, "HVTWZ_2600":302259, "HVTWZ_3000":302261, "HVTWZ_3500":302262, "HVTWZ_4000":302263, "HVTWZ_4500":302264, "HVTWZ_5000":302265, "HVTWZVBF_500":307705, "HVTWZVBF_600":307706, "HVTWZVBF_700":307707, "HVTWZVBF_800":307708, "HVTWZVBF_1000":307710, "HVTWZVBF_1200":307712, "HVTWZVBF_1500":307715, "HVTWZVBF_1800":307718, "HVTWZVBF_2000":307720, "HVTWZVBF_2400":307722, "HVTWZVBF_2600":307723, "HVTWZVBF_3000":307725, "HVTWZVBF_3500":307726, "HVTWZVBF_4000":307727, "Rad_300":310039, "Rad_700":310040, "Rad_1000":310041, "Rad_2000":310042, "Rad_3000":310043, "Rad_4000":310044, "Rad_5000":310045, "Rad_6000":310046, "RadVBF_300":310047, "RadVBF_700":310048, "RadVBF_1000":310049, "RadVBF_2000":310050, "RadVBF_3000":310051, "RadVBF_4000":310052, "RadVBF_5000":310053, "RadVBF_6000":310054, "RSG_200":307442, "RSG_300":307443, "RSG_400":307444, "RSG_600":303303, "RSG_700":303304, "RSG_800":303305, "RSG_1000":303307, "RSG_1200":303309, "RSG_1500":303312, "RSG_1800":303315, "RSG_2000":303317, "RSG_2400":303319, "RSG_2600":303320, "RSG_3000":303322, "RSG_3500":303323, "RSG_4000":303324, "RSG_4500":303325, "RSG_5000":303326, "RSGVBF_300":310031, "RSGVBF_700":310032, "RSGVBF_1000":310033, "RSGVBF_2000":310034, "RSGVBF_3000":310035, "RSGVBF_4000":310036, "RSGVBF_5000":310037, "RSGVBF_6000":310038 }
  if "L1" in region:
    dsidmap={"HVTWW_300":307365, "HVTWW_400":307366, "HVTWW_500":302116, "HVTWW_600":302117, "HVTWW_700":302118, "HVTWW_800":302119, "HVTWW_1000":302121, "HVTWW_1200":302123, "HVTWW_1400":302125, "HVTWW_1500":302126, "HVTWW_1600":302127, "HVTWW_1800":302129, "HVTWW_2000":302131, "HVTWW_2400":302133, "HVTWW_2600":302134, "HVTWW_3000":302136, "HVTWW_3500":302137, "HVTWW_4000":302138, "HVTWW_4500":302139, "HVTWW_5000":302140, "HVTWWVBF_300":307563, "HVTWWVBF_400":307564, "HVTWWVBF_600":307566, "HVTWWVBF_700":307567, "HVTWWVBF_1000":307570, "HVTWWVBF_1200":307572, "HVTWWVBF_1500":307575, "HVTWWVBF_1800":307578, "HVTWWVBF_2000":307580, "HVTWWVBF_2400":307582, "HVTWWVBF_2600":307583, "HVTWWVBF_3000":307585, "HVTWWVBF_3500":307586, "HVTWWVBF_4000":307587, "HVTWZ_300":307374, "HVTWZ_400":307375, "HVTWZ_500":302191, "HVTWZ_600":302192, "HVTWZ_700":302193, "HVTWZ_800":302194, "HVTWZ_1000":302196, "HVTWZ_1200":302198, "HVTWZ_1400":302200, "HVTWZ_1500":302201, "HVTWZ_1600":302202, "HVTWZ_1800":302204, "HVTWZ_2000":302206, "HVTWZ_2400":302208, "HVTWZ_2600":302209, "HVTWZ_3000":302211, "HVTWZ_3500":302212, "HVTWZ_4000":302213, "HVTWZ_4500":302214, "HVTWZ_5000":302215, "HVTWZVBF_300":307647, "HVTWZVBF_400":307648, "HVTWZVBF_600":307650, "HVTWZVBF_700":307651, "HVTWZVBF_800":307652, "HVTWZVBF_1000":307654, "HVTWZVBF_1200":307656, "HVTWZVBF_1500":307659, "HVTWZVBF_1800":307662, "HVTWZVBF_2000":307664, "HVTWZVBF_2400":307666, "HVTWZVBF_2600":307667, "HVTWZVBF_3000":307669, "HVTWZVBF_3500":307670, "HVTWZVBF_4000":307671, "Rad_300":310015, "Rad_700":310016, "Rad_1000":310017, "Rad_2000":310018, "Rad_3000":310019, "Rad_4000":310020, "Rad_5000":310021, "Rad_6000":310022, "RadVBF_300":310023, "RadVBF_700":310024, "RadVBF_1000":310025, "RadVBF_2000":310026, "RadVBF_3000":310027, "RadVBF_4000":310028, "RadVBF_5000":310029, "RadVBF_6000":310030, "RSG_300":307474, "RSG_400":307475, "RSG_600":303225, "RSG_700":303226, "RSG_800":303227, "RSG_1000":303229, "RSG_1200":303231, "RSG_1500":303234, "RSG_1800":303237, "RSG_2000":303239, "RSG_2400":303241, "RSG_2600":303242, "RSG_3000":303244, "RSG_3500":303245, "RSG_4000":303246, "RSG_4500":303247, "RSG_5000":303248, "RSGVBF_300":310007, "RSGVBF_700":310008, "RSGVBF_1000":310009, "RSGVBF_2000":310010, "RSGVBF_3000":310011, "RSGVBF_4000":310012, "RSGVBF_5000":310013, "RSGVBF_6000":310014}
  if "L2" in region:
    '''
    dsidmap={"HVTWZ_200":307360, "HVTWZ_400":307362, "HVTWZ_500":302216, "HVTWZ_600":302217, "HVTWZ_700":302218, "HVTWZ_800":302219, "HVTWZ_1000":302221, "HVTWZ_1200":302223, "HVTWZ_1400":302225, "HVTWZ_1500":302226, "HVTWZ_1600":302227, "HVTWZ_1800":302229, "HVTWZ_2000":302231, "HVTWZ_2400":302233, "HVTWZ_2600":302234, "HVTWZ_3000":302236, "HVTWZ_3500":302237, "HVTWZ_4000":302238, "HVTWZ_4500":302239, "HVTWZ_5000":302240, "HVTWZVBF_250":307674, "HVTWZVBF_300":307675, "HVTWZVBF_400":307676, "HVTWZVBF_600":307678, "HVTWZVBF_700":307679, "HVTWZVBF_800":307680, "HVTWZVBF_1000":307682, "HVTWZVBF_1200":307684, "HVTWZVBF_1500":307687, "HVTWZVBF_1800":307690, "HVTWZVBF_2000":307692, "HVTWZVBF_2400":307694, "HVTWZVBF_2600":307695, "HVTWZVBF_3000":307697, "HVTWZVBF_3500":307698, "HVTWZVBF_4000":307699, "Rad_300":309991, "Rad_700":309992, "Rad_1000":309993, "Rad_2000":309994, "Rad_3000":309995, "Rad_4000":309996, "Rad_5000":309997, "Rad_6000":309998, "RadVBF_300":309999, "RadVBF_700":310000, "RadVBF_1000":310001, "RadVBF_2000":310002, "RadVBF_3000":310003, "RadVBF_4000":310004, "RadVBF_5000":310005, "RadVBF_6000":310006, "RSG_200":307476, "RSG_300":307477, "RSG_400":307478, "RSG_600":303278, "RSG_700":303279, "RSG_800":303280, "RSG_1000":303282, "RSG_1200":303284, "RSG_1500":303287, "RSG_1800":303290, "RSG_2000":303292, "RSG_2400":303294, "RSG_2600":303295, "RSG_3000":303297, "RSG_3500":303298, "RSG_4000":303299, "RSG_4500":303300, "RSG_5000":303301, "RSGVBF_300":309983, "RSGVBF_700":309984, "RSGVBF_1000":309985, "RSGVBF_2000":309986, "RSGVBF_3000":309987, "RSGVBF_4000":309988, "RSGVBF_5000":309989, "RSGVBF_6000":309990}
    '''
    dsidmap={"HVTWZ_200":307360, "HVTWZ_400":307362, "HVTWZ_500":302216, "HVTWZ_600":302217, "HVTWZ_700":302218, "HVTWZ_800":302219, "HVTWZ_1000":302221, "HVTWZ_1200":302223, "HVTWZ_1400":302225, "HVTWZ_1500":302226, "HVTWZ_1600":302227, "HVTWZ_1800":302229, "HVTWZ_2000":302231, "HVTWZ_2400":302233, "HVTWZ_2600":302234, "HVTWZ_3000":302236, "HVTWZ_3500":302237, "HVTWZ_4000":302238, "HVTWZ_4500":302239, "HVTWZ_5000":302240, "HVTWZVBF_250":307674, "HVTWZVBF_300":307675, "HVTWZVBF_400":307676, "HVTWZVBF_600":307678, "HVTWZVBF_700":307679, "HVTWZVBF_800":307680, "HVTWZVBF_1000":307682, "HVTWZVBF_1200":307684, "HVTWZVBF_1500":307687, "HVTWZVBF_1800":307690, "HVTWZVBF_2000":307692, "HVTWZVBF_2400":307694, "HVTWZVBF_2600":307695, "HVTWZVBF_3000":307697, "HVTWZVBF_3500":307698, "HVTWZVBF_4000":307699,
             "Radion_300":451141,
             "Radion_400":451142,
             "Radion_500":451143,
             "Radion_600":451144,
             "Radion_700":451145,
             "Radion_800":451146,
             "Radion_1000":451147,
             "Radion_1200":451148,
             "Radion_1400":451149,
             "Radion_1500":451150,
             "Radion_1600":451151,
             "Radion_1800":451152,
             "Radion_2000":451153,
             "Radion_2400":451154,
             "Radion_2600":451155,
             "Radion_3000":451156,
             "Radion_3500":451157,
             "Radion_4000":451158,
             "Radion_4500":451159,
             "Radion_5000":451160,
             "Radion_6000":451161,
             "RadVBF_300":309999, "RadVBF_700":310000, "RadVBF_1000":310001, "RadVBF_2000":310002, "RadVBF_3000":310003, "RadVBF_4000":310004, "RadVBF_5000":310005, "RadVBF_6000":310006, "RSG_200":307476, "RSG_300":307477, "RSG_400":307478, "RSG_600":303278, "RSG_700":303279, "RSG_800":303280, "RSG_1000":303282, "RSG_1200":303284, "RSG_1500":303287, "RSG_1800":303290, "RSG_2000":303292, "RSG_2400":303294, "RSG_2600":303295, "RSG_3000":303297, "RSG_3500":303298, "RSG_4000":303299, "RSG_4500":303300, "RSG_5000":303301, "RSGVBF_300":309983, "RSGVBF_700":309984, "RSGVBF_1000":309985, "RSGVBF_2000":309986, "RSGVBF_3000":309987, "RSGVBF_4000":309988, "RSGVBF_5000":309989, "RSGVBF_6000":309990}
    
  if signal+"_"+str(mass) in dsidmap:
    print("DSID found is ",str(dsidmap[signal+"_"+str(mass)])) 
    return str(dsidmap[signal+"_"+str(mass)])
  else:
    return 0

if __name__ == "__main__":

  ####################################################
  # Preamble to setup the RF implmentation and options form the command line
  ####################################################

  #Example of how to grab an option from the command line
  tag="test200IntNoteArch"
  # EJS, 2021-10-29
  # Here I'm putting the Reader's output obtained by running Alessandra's pDNN (first training with loose pDNN configuration)
  # Gabriele needs these files to do tests on the pDNN scores directly from the Nominal tree
  # basedirectory="/eos/user/s/schioppa/analysis/development/2021-10-21_AlessandraTest/ntuples/ntuples/" 
  # basedirectory="/nfs/kloe/einstein4/HDBS/PDNNTestAGS/ntuples-old" # as before but copied on Lecce 
  ### same processing as before but re-run to let all job complete successfully (recover some statistics)
  #basedirectory="/nfs/kloe/einstein4/HDBS/PDNNTestAGS/ntuples" # as before but from Lecce
  #basedirectory="/nfs/kloe/einstein4/HDBS/ReaderOutputWithScores/reader_mc16a_VV_2lep_PFlow_UFO_withScores/fetch/data-MVATree" #reader on r33-22, scores from  Radion_ggF_Merged_ATL-COM-PHYS-2018-1549
  basedirectory="/nfs/kloe/einstein4/stefania/CxAODReaderProd/tmp" #link to previous one with Wjets-0.root and data removed 
  

  usePNN=False
  use0L=False
  use1L=False
  use2L=False
  useMerg=False
  useRes=False
  useSR=False
  useZCR=False
  useWCR=False
  useTCR=False
  useData=False
  
  print("Running macro: ", sys.argv[0])

  
  if len(sys.argv) > 1:
    if "0" in sys.argv[1]: use0L=True
    if "1" in sys.argv[1]: use1L=True
    if "2" in sys.argv[1]: use2L=True
    if "SR" in sys.argv[1]: useSR=True
    if "WCR" in sys.argv[1]: useWCR=True
    if "ZCR" in sys.argv[1]: useZCR=True
    if "TCR" in sys.argv[1]: useTCR=True
    if "Merg" in sys.argv[1]: useMerg=True
    if "Res" in sys.argv[1]: useRes=True
    if "PNN" in sys.argv[1]: usePNN=True
  if usePNN: tag=tag+"PNN"

  masses=[]
  if len(sys.argv) > 2:
    masses=[int(sys.argv[2])]

  ### for /nfs/kloe/einstein4/HDBS/ReaderOutputWithScores/reader_mc16a_VV_2lep_PFlow_UFO_withScores/
  ### vector<int> PDNNMassHypothesis = 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 4100 4200 4300 4400 4500 4600 4700 4800 4900 5000 5100 5200 5300 5400 5500 5600 5700 5800 5900 6000
  PDNNMassHypothesis = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000]
  massIndex=[]
  massIndex=[PDNNMassHypothesis.index(masses[0])]
  #for i in range(len(masses)):
  #  massIndex[i] = PDNNMassHypothesis.index(masses[i])


  print("Mass : "+ str(masses[0])+" Mass index :"+str(massIndex[0]))


  #exit()  
    


  signalType="HVTWZ"
  if len(sys.argv) > 3:
    signalType=sys.argv[3]

  print("SignalType is ", signalType)  
    
  #Start analysis and define directory structure
  RFAnalysis = RF.DefaultTreeAnalysisRunner("ExampleVV")
  #RFAnalysis.setOutputDir("./output/ZZSignalRegionsGGF")
  RFAnalysis.setOutputDir("/scratch/spagnolo/RF2021/ZZSRMergGGF_"+tag)
  RFAnalysis.setOutputWSTag(tag)

  #Set up/down variation names to your convention
  RFAnalysis.setCollectionTagNominal("Nominal")
  RFAnalysis.setCollectionTagUp("_up")
  RFAnalysis.setCollectionTagDown("_dn")

  #Example of defining some variables ahead of time. Example here for lumi uncertainities
  lumi_scale=1.
  lumi_uncert=0.017

  ####################################################
  # Now define your fit model and where to find the data
  ####################################################
  regions= {} #region=[useRealData, regionFlag, binning, observableName, weightName]
  if use0L:
    if useSR: 
      if useMerg:
        ####### to be revised ... most probably wrong 
        regions["L0_MergHP_GGF_ZZ_SR"]=[useData, "Pass_MergHP_GGF_ZZ_SR", findBinning("L0_MergHP_GGF_ZZ_SR",usePNN), "X_boosted_m", "weight"]
        regions["L0_MergLP_GGF_ZZ_SR"]=[useData, "Pass_MergLP_GGF_ZZ_SR", findBinning("L0_MergLP_GGF_ZZ_SR",usePNN), "X_boosted_m", "weight"]
  if use1L:
    if useSR: 
      if useMerg:
        ####### to be revised ... certainly wrong 
        regions["L1_MergHP_GGF_ZZ_Untag_SR"]=[useData, "Pass_MergHP_GGF_ZZ_Untag_SR", findBinning("L1_MergHP_GGF_ZZ_SR",usePNN), "X_boosted_m", "weight"]
        regions["L1_MergHP_GGF_ZZ_Tag_SR"]=[useData, "Pass_MergHP_GGF_ZZ_Untag_SR", findBinning("L1_MergHP_GGF_ZZ_SR",usePNN), "X_boosted_m", "weight"]
        regions["L1_MergLP_GGF_ZZ_Untag_SR"]=[useData, "Pass_MergLP_GGF_ZZ_Untag_SR", findBinning("L1_MergLP_GGF_ZZ_SR",usePNN), "X_boosted_m", "weight"]
        regions["L1_MergLP_GGF_ZZ_Tag_SR"]=[useData, "Pass_MergLP_GGF_ZZ_Untag_SR", findBinning("L1_MergLP_GGF_ZZ_SR",usePNN), "X_boosted_m", "weight"]
      if useRes:
        ####### to be revised ... certainly wrong 
        regions["L1_Resolved_GGF_ZZ_Untag_SR"]=[useData, "Pass_Resolved_GGF_ZZ_Untag_SR", findBinning("L1_Resolved_GGF_ZZ_SR",usePNN), "X_resolved_ZZ_m", "weight"]
        regions["L1_Resolved_GGF_ZZ_Tag_SR"]=[useData, "Pass_Resolved_GGF_ZZ_Untag_SR", findBinning("L1_Resolved_GGF_ZZ_SR",usePNN), "X_resolved_ZZ_m", "weight"]
    if useTCR:
      if useMerg:
        ####### to be revised ... certainly wrong 
        regins["L1_MergHP_GGF_ZZ_Untag_TCR"]=[useData, "Pass_MergHP_GGF_ZZ_Untag_TCR", findBinning("L1_MergHP_GGF_ZZ_TCR",usePNN), "X_boosted_m", "weight"]
        regions["L1_MergHP_GGF_ZZ_Tag_TCR"]=[useData, "Pass_MergHP_GGF_ZZ_Untag_TCR", findBinning("L1_MergHP_GGF_ZZ_TCR",usePNN), "X_boosted_m", "weight"]
        regions["L1_MergLP_GGF_ZZ_Untag_TCR"]=[useData, "Pass_MergLP_GGF_ZZ_Untag_TCR", findBinning("L1_MergLP_GGF_ZZ_TCR",usePNN), "X_boosted_m", "weight"]
        regions["L1_MergLP_GGF_ZZ_Tag_TCR"]=[useData, "Pass_MergLP_GGF_ZZ_Untag_TCR", findBinning("L1_MergLP_GGF_ZZ_TCR",usePNN), "X_boosted_m", "weight"]
      if useRes:
        ####### to be revised ... certainly wrong 
        regions["L1_Resolved_GGF_ZZ_Untag_TCR"]=[useData, "Pass_Resolved_GGF_ZZ_Untag_TCR", findBinning("L1_Resolved_GGF_ZZ_TCR",usePNN), "X_resolved_ZZ_m", "weight"]
        regions["L1_Resolved_GGF_ZZ_Tag_TCR"]=[useData, "Pass_Resolved_GGF_ZZ_Untag_TCR", findBinning("L1_Resolved_GGF_ZZ_TCR",usePNN), "X_resolved_ZZ_m", "weight"]
    if useWCR:
      if useMerg:
        ####### to be revised ... certainly wrong 
        regions["L1_MergHP_GGF_ZZ_Untag_WCR"]=[useData, "Pass_MergHP_GGF_ZZ_Untag_WCR", findBinning("L1_MergHP_GGF_ZZ_WCR",usePNN), "X_boosted_m", "weight"]
        regions["L1_MergHP_GGF_ZZ_Tag_WCR"]=[useData, "Pass_MergHP_GGF_ZZ_Untag_WCR", findBinning("L1_MergHP_GGF_ZZ_WCR",usePNN), "X_boosted_m", "weight"]
        regions["L1_MergLP_GGF_ZZ_Untag_WCR"]=[useData, "Pass_MergLP_GGF_ZZ_Untag_WCR", findBinning("L1_MergLP_GGF_ZZ_WCR",usePNN), "X_boosted_m", "weight"]
        regions["L1_MergLP_GGF_ZZ_Tag_WCR"]=[useData, "Pass_MergLP_GGF_ZZ_Untag_WCR", findBinning("L1_MergLP_GGF_ZZ_WCR",usePNN), "X_boosted_m", "weight"]
      if useRes:
        regions["L1_Resolved_GGF_ZZ_Untag_WCR"]=[useData, "Pass_Resolved_GGF_ZZ_Untag_WCR", findBinning("L1_Resolved_GGF_ZZ_WCR",usePNN), "X_resolved_ZZ_m", "weight"]
        regions["L1_Resolved_GGF_ZZ_Tag_WCR"]=[useData, "Pass_Resolved_GGF_ZZ_Untag_WCR", findBinning("L1_Resolved_GGF_ZZ_WCR",usePNN), "X_resolved_ZZ_m", "weight"]
  if use2L:
    if useSR: 
      if useMerg:
        regions["L2_MergHP_GGF_ZZ_Untag_SR"]=[useData, "Pass_MergHP_GGF_ZZ_Untag_SR", findBinning("L2_MergHP_GGF_ZZ_Untag_SR",usePNN), "X_boosted_m", "weight"]
        regions["L2_MergLP_GGF_ZZ_Untag_SR"]=[useData, "Pass_MergLP_GGF_ZZ_Untag_SR", findBinning("L2_MergLP_GGF_ZZ_Untag_SR",usePNN), "X_boosted_m", "weight"]
        regions["L2_MergHP_GGF_ZZ_Tag_SR"  ]=[useData, "Pass_MergHP_GGF_ZZ_Tag_SR",   findBinning("L2_MergHP_GGF_ZZ_Tag_SR",  usePNN), "X_boosted_m", "weight"]
        regions["L2_MergLP_GGF_ZZ_Tag_SR"  ]=[useData, "Pass_MergLP_GGF_ZZ_Tag_SR",   findBinning("L2_MergLP_GGF_ZZ_Tag_SR",  usePNN), "X_boosted_m", "weight"]
      if useRes:
        regions["L2_Resolved_GGF_ZZ_Untag_SR"]=[useData, "Pass_Resolved_GGF_ZZ_Untag_SR", findBinning("L2_Resolved_GGF_ZZ_Untag_SR",usePNN), "X_resolved_WZ_m", "weight"]
        regions["L2_Resolved_GGF_ZZ_Tag_SR"  ]=[useData, "Pass_Resolved_GGF_ZZ_Tag_SR",   findBinning("L2_Resolved_GGF_ZZ_Tag_SR"  ,usePNN), "X_resolved_WZ_m", "weight"]
    if useZCR:
      if useMerg:
        regions["L2_MergHP_GGF_ZZ_Untag_ZCR"]=[useData, "Pass_MergHP_GGF_ZZ_Untag_ZCR", findBinning("L2_MergHP_GGF_ZZ_Untag_ZCR",usePNN), "X_boosted_m", "weight"]
        regions["L2_MergLP_GGF_ZZ_Untag_ZCR"]=[useData, "Pass_MergLP_GGF_ZZ_Untag_ZCR", findBinning("L2_MergLP_GGF_ZZ_Untag_ZCR",usePNN), "X_boosted_m", "weight"]
        regions["L2_MergHP_GGF_ZZ_Untag_ZCR"]=[useData, "Pass_MergHP_GGF_ZZ_Tag_ZCR",   findBinning("L2_MergHP_GGF_ZZ_Tag_ZCR",  usePNN), "X_boosted_m", "weight"]
        regions["L2_MergLP_GGF_ZZ_Untag_ZCR"]=[useData, "Pass_MergLP_GGF_ZZ_Tag_ZCR",   findBinning("L2_MergLP_GGF_ZZ_Tag_ZCR",  usePNN), "X_boosted_m", "weight"]
      if useRes:
        regions["L2_Resolved_GGF_ZZ_Untag_ZCR"]=[useData, "Pass_Resolved_GGF_ZZ_Untag_ZCR", findBinning("L2_Resolved_GGF_ZZ_Untag_ZCR",usePNN), "X_resolved_WZ_m", "weight"]
        regions["L2_Resolved_GGF_ZZ_Tag_ZCR"]  =[useData, "Pass_Resolved_GGF_ZZ_Tag_ZCR",   findBinning("L2_Resolved_GGF_ZZ_Tag_ZCR",  usePNN), "X_resolved_WZ_m", "weight"]

  if usePNN:
    # override the content of the 4th variable or the regions array (i.e. the variable to be used in the fit)
    for region in regions:
      #regions[region][3]="pDNNScore"+str(masses[0])
      regions[region][3]="pDNNScore"+str(massIndex[0])

  print("Variable to use is  <"+regions[region][3]+">")


  samples = {
      # sample   : [ path, filter, name, minmu , maxmu ,constrianType]
      "Diboson"  : [ "Diboson-*.root", "", "XS_Diboson", 0.94, 1.06, RF.MultiplicativeFactor.GAUSSIAN],
      "stop"     : [ "stop-*.root", "", "XS_stop", 0.80, 1.20, RF.MultiplicativeFactor.GAUSSIAN],
    }
  if useZCR: samples["Zjets"]=[ "Zjets-*.root", "", "XS_Zjets", 0.5, 1.5, RF.MultiplicativeFactor.FREE]
  else: samples["Zjets"]=[ "Zjets-*.root", "", "XS_Zjets", 0.8, 1.2, RF.MultiplicativeFactor.GAUSSIAN]
  if useWCR: samples["Wjets"]=[ "Wjets-*.root", "", "XS_Wjets", 0.5, 1.5, RF.MultiplicativeFactor.FREE]
  else: samples["Wjets"]=[ "Wjets-*.root", "", "XS_Wjets", 0.8, 1.2, RF.MultiplicativeFactor.GAUSSIAN]
  if useTCR: samples["ttbar"]=[ "ttbar-*.root", "", "XS_ttbar", 0.5, 1.5, RF.MultiplicativeFactor.FREE]
  else: samples["ttbar"]=[ "ttbar-*.root", "", "XS_ttbar", 0.8, 1.2, RF.MultiplicativeFactor.GAUSSIAN]

  signals = {} #signal=[path, signalType, mass]
  for mass in masses:
    signals[signalType+"_"+str(mass)]=[signalType+"-*.root",signalType,mass]

  variations=[]

  #Loop over regions
  for region in regions:
    #path=basedirectory+"/inputVV_"+str(regions[region][2])+"lep/fetch/"
    path=basedirectory+"/"

    #Add channel (aka a region) to model
    print("Adding channel: ", region)
    binInfo=array.array('d',regions[region][2])
    RFAnalysis.addChannel(region,len(regions[region][2])-1,binInfo,regions[region][3],regions[region][4],regions[region][1])

    #Add data to the channel, or apply fake data
    if regions[region][0]==0:
      RFAnalysis.fakeData(region)
    else:
      RFAnalysis.channel(region).addData("data","data-*.root")

    #Add signals, if define them with different name will be seperate workspaces
    for sigName in signals:
      signalInfo=signals[sigName]
      RFAnalysis.channel(region).addSample(sigName,path+signalInfo[0],"DSID=="+str(findDSID(region,signalInfo[1],signalInfo[2])))
      RFAnalysis.channel(region).sample(sigName).multiplyBy("mu", 0, 0, 1000)
      RFAnalysis.channel(region).sample(sigName).multiplyBy("lumiNP",   lumi_scale, lumi_scale*(1-lumi_uncert), lumi_scale*(1.+lumi_uncert),RF.MultiplicativeFactor.GAUSSIAN) 
      RFAnalysis.channel(region).sample(sigName).setUseStatError(True)
      RFAnalysis.defineSignal(sigName,sigName) #Let RF know this sample is a signal, can have multiple signal

    #Add Background signal to the channel
    for sampleName in samples:
      sampleInfo=samples[sampleName]
      RFAnalysis.channel(region).addSample(sampleName,path+sampleInfo[0]) 
      RFAnalysis.channel(region).sample(sampleName).multiplyBy(sampleInfo[2], 1.0, sampleInfo[3],sampleInfo[4],sampleInfo[5])
      RFAnalysis.channel(region).sample(sampleName).multiplyBy("lumiNP",   lumi_scale, lumi_scale*(1-lumi_uncert), lumi_scale*(1.+lumi_uncert),RF.MultiplicativeFactor.GAUSSIAN)
      RFAnalysis.channel(region).sample(sampleName).setUseStatError(True)
      for var in variations:
        RFAnalysis.channel(region).sample(sampleName).addVariation(var)

  ####################################################
  # Finally define last few things and build workspace
  ####################################################
  # define POI
  RFAnalysis.setPOI("mu")

  #make workspace
  RFAnalysis.produceWS()
