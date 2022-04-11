#include "chains/chainSet_33_22_pdnn.h"
//#include "initChains_33_22_pdnn.C"
#include "makeMyQuickPlot2022_pdnn.C"

void main_QuickPlot2022_pdnn()
{
  // instantiate a chainSet 
  chainSet_33_22_pdnn* cs = new chainSet_33_22_pdnn();

  std::string tag="Run2";
  bool useW=true; bool shapeComp=false;
  bool savePlots=true;
  bool logx=false; bool logy=true;
  std::string scoreSignal="";
  std::string userCut="";
  
  
  ///////// ZZ  VBF SR and ZCR ----------------------------------
  // Merged HP, Merged LP, Resolved 
  ///////// ZZ  ggF/DY SR and ZCR ----------------------------------	
  // Merged HP Tag, Untag,  LP Tag, Untag, Resolved  LP Tag, Untag
  std::string userCutArrayForZZ_SR[9] = {"Pass_MergHP_VBF_ZZ_SR==1",  "Pass_MergLP_VBF_ZZ_SR==1",  "Pass_Res_VBF_ZZ_SR==1",
                                         "Pass_MergHP_GGF_ZZ_Tag_SR==1",  "Pass_MergHP_GGF_ZZ_Untag_SR==1",
					 "Pass_MergLP_GGF_ZZ_Tag_SR==1",  "Pass_MergLP_GGF_ZZ_Untag_SR==1",
					 "Pass_Res_GGF_ZZ_Tag_SR==1"   ,  "Pass_Res_GGF_ZZ_Untag_SR==1"   };
  std::string userCutArrayForZZ_ZCR[9]= {"Pass_MergHP_VBF_ZZ_ZCR==1",  "Pass_MergLP_VBF_ZZ_ZCR==1",  "Pass_Res_VBF_ZZ_ZCR==1",
                                         "Pass_MergHP_GGF_ZZ_Tag_ZCR==1",  "Pass_MergHP_GGF_ZZ_Untag_ZCR==1",
					 "Pass_MergLP_GGF_ZZ_Tag_ZCR==1",  "Pass_MergLP_GGF_ZZ_Untag_ZCR==1",
					 "Pass_Res_GGF_ZZ_Tag_ZCR==1"   ,  "Pass_Res_GGF_ZZ_Untag_ZCR==1"   };
  ///////// WZ  VBF SR and ZCR ----------------------------------
  // Merged HP, Merged LP, Resolved  
  ///////// WZ  ggF/DY SR and ZCR ----------------------------------
  // Merged HP, Merged LP, Resolved  
  std::string userCutArrayForWZ_SR[6] = {"Pass_MergHP_VBF_WZ_SR==1",  "Pass_MergLP_VBF_WZ_SR==1",  "Pass_Res_VBF_WZ_SR==1",  "Pass_MergHP_GGF_WZ_SR==1",  "Pass_MergLP_GGF_WZ_SR==1",  "Pass_Res_GGF_WZ_SR==1" };
  std::string userCutArrayForWZ_ZCR[6]= {"Pass_MergHP_VBF_WZ_ZCR==1", "Pass_MergLP_VBF_WZ_ZCR==1", "Pass_Res_VBF_WZ_ZCR==1", "Pass_MergHP_GGF_WZ_ZCR==1", "Pass_MergLP_GGF_WZ_ZCR==1", "Pass_Res_GGF_WZ_ZCR==1"};


  //std::string myvarArray[4]={"pDNNScore600","pDNNScore800","pDNNScore1000", "pDNNScore3000"};
  std::string myvarArray[2]={"pDNNScore600","pDNNScore3000"};
  //int nBins=100;double xmin=0.;double xmax=1.; 
  
  bool showMass=true;
  int nBins=100;double xmin=0.;double xmax=4000.; 
  std::string myvar="";
  
  //  for (unsigned int j=0; j<2; ++j) 
  //   {
  //     myvar = myvarArray[j]; 
  scoreSignal="RSG";	
  for (unsigned int i=0; i<9; ++i)
    {	
      userCut=userCutArrayForZZ_SR[i];
      if (userCut.substr(0,9)=="Pass_Merg" && showMass) myvar="X_boosted_m";
      else if (showMass) myvar="X_resolved_ZZ_m";

      makeMyQuickPlot2022_pdnn(myvar, nBins, xmin, xmax, logx, logy, cs, userCut, tag, useW, shapeComp, scoreSignal, savePlots);
      userCut=userCutArrayForZZ_ZCR[i];
      makeMyQuickPlot2022_pdnn(myvar, nBins, xmin, xmax, logx, logy, cs, userCut, tag, useW, shapeComp, scoreSignal, savePlots);
    }
  scoreSignal="Radion";
  for (unsigned int i=0; i<9; ++i)
    {
      userCut=userCutArrayForZZ_SR[i];
      if (userCut.substr(0,9)=="Pass_Merg" && showMass) myvar="X_boosted_m";
      else if (showMass) myvar="X_resolved_ZZ_m";
      makeMyQuickPlot2022_pdnn(myvar, nBins, xmin, xmax, logx, logy, cs, userCut, tag, useW, shapeComp, scoreSignal, savePlots);
      userCut=userCutArrayForZZ_ZCR[i];
      makeMyQuickPlot2022_pdnn(myvar, nBins, xmin, xmax, logx, logy, cs, userCut, tag, useW, shapeComp, scoreSignal, savePlots);
    }
  
  scoreSignal="HVTWZ";
  for (unsigned int i=0; i<6; ++i)
    {
      userCut=userCutArrayForWZ_SR[i];
      if (userCut.substr(0,9)=="Pass_Merg" && showMass) myvar="X_boosted_m";
      else if (showMass) myvar="X_resolved_WZ_m";
      makeMyQuickPlot2022_pdnn(myvar, nBins, xmin, xmax, logx, logy, cs, userCut, tag, useW, shapeComp, scoreSignal, savePlots);
      userCut=userCutArrayForWZ_ZCR[i];
      makeMyQuickPlot2022_pdnn(myvar, nBins, xmin, xmax, logx, logy, cs, userCut, tag, useW, shapeComp, scoreSignal, savePlots);
    }

  //    }
  

}
