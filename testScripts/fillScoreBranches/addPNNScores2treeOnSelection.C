// Load the library at macro parsing time: we need this to use its content in the code
/// R__LOAD_LIBRARY(/test/libEvent.so)

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "TFile.h"
#include "TH1F.h"
#include "TTree.h"
#include "TFile.h"
#include "TCanvas.h"

//void addPNNscores2tree(std::string fName,  std::string analysis="merged", std::string prodChannel="ggF", std::string signal = "RSG")
void addPNNScores2treeOnSelection(std::string fName,
				  std::string analysis   ="merged",
				  std::string prodChannel="ggF",
				  std::string signal     = "RSG",
				  std::string mcType     = "mc16a",
				  std::string tag        =""        )
{
  std::string toErase=".root";
  size_t pos = fName.find(toErase);  
  if (pos != std::string::npos)
    {      
      // If found then erase it from string
      fName.erase(pos, toErase.length());
    }  

  std::string inputPath="/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_"+mcType+"_VV_2lep_PFlow_UFO/fetch/data-MVATree/";
  //std::string inputPathScores="/scratch/stefania/PNNScores/";
  //std::string outputPath="/scratch/stefania/rootPNNScores/";
  std::string inputPathScores=tag;
  std::string outputPath=tag;
  // input scores tag 
   std::string inputScoreFileN = "Scores_"+fName+"_"+analysis+"_"+prodChannel+"_"+signal;
  // input tree 
   std::string filename = inputPath+fName+".root";
   //output tree 
   std::string outfilename = outputPath+"/"+inputScoreFileN+".root";
   // add scores file 
   std::string fn = inputPathScores+"/"+inputScoreFileN+".txt"; 
   //TString filename = "/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16e_VV_2lep_PFlow_UFO/fetch/data-MVATree/data18-0.root";
 
   TFile oldfile(filename.c_str());
   TTree *oldtree;
   oldfile.GetObject("Nominal", oldtree);
   std::cout<<"Tree read from file"<<std::endl;
   long nentries = oldtree->GetEntries();
   std::cout<<"N. of events = "<<nentries<<std::endl;
 
   // Deactivate all branches
   // oldtree->SetBranchStatus("*", 0);
 
   // Activate only four of them
   //for (auto activeBranchName : {"event", "fNtrack", "fNseg", "fH"})
   //   oldtree->SetBranchStatus(activeBranchName, 1);

   
   // Create a new file + a clone of old tree in new file
   TFile newfile(outfilename.c_str(), "recreate");
   auto newtree = oldtree->CloneTree();
   std::cout<<"Tree cloned"<<std::endl;
   

   newtree->SetName("Nominal");


   float pDNNscArray[61]={0.};
   unsigned int i=0;
   float *pDNNScore500    = &pDNNscArray[i];++i;// 0;
   float *pDNNScore600    = &pDNNscArray[i];++i;//=0;
   float *pDNNScore700    = &pDNNscArray[i];++i;//=0;
   float *pDNNScore800    = &pDNNscArray[i];++i;//=0;
   float *pDNNScore900    = &pDNNscArray[i];++i;//=0;
   float *pDNNScore1000   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore1100   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore1200   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore1300   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore1400   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore1500   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore1600   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore1700   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore1800   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore1900   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore2000   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore2100   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore2200   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore2300   = &pDNNscArray[i];++i;//=0;
   float *pDNNScore2400   = &pDNNscArray[i];++i;//0;
   float *pDNNScore2500   = &pDNNscArray[i];++i;//0;
   float *pDNNScore2600   = &pDNNscArray[i];++i;//0;
   float *pDNNScore2700   = &pDNNscArray[i];++i;//0;
   float *pDNNScore2800   = &pDNNscArray[i];++i;//0;
   float *pDNNScore2900   = &pDNNscArray[i];++i;//0;
   float *pDNNScore3000   = &pDNNscArray[i];++i;//0;
   float *pDNNScore3100   = &pDNNscArray[i];++i;//0;
   float *pDNNScore3200   = &pDNNscArray[i];++i;//0;
   float *pDNNScore3300   = &pDNNscArray[i];++i;//0;
   float *pDNNScore3400   = &pDNNscArray[i];++i;//0;
   float *pDNNScore3500   = &pDNNscArray[i];++i;//0;
   float *pDNNScore3600   = &pDNNscArray[i];++i;//0;
   float *pDNNScore3700   = &pDNNscArray[i];++i;//0;
   float *pDNNScore3800   = &pDNNscArray[i];++i;//0;
   float *pDNNScore3900   = &pDNNscArray[i];++i;//0;
   float *pDNNScore4000   = &pDNNscArray[i];++i;//0;
   float *pDNNScore4100   = &pDNNscArray[i];++i;//0;
   float *pDNNScore4200   = &pDNNscArray[i];++i;//0;
   float *pDNNScore4300   = &pDNNscArray[i];++i;//0;
   float *pDNNScore4400   = &pDNNscArray[i];++i;//0;
   float *pDNNScore4500   = &pDNNscArray[i];++i;//0;
   float *pDNNScore4600   = &pDNNscArray[i];++i;//0;
   float *pDNNScore4700   = &pDNNscArray[i];++i;//0;
   float *pDNNScore4800   = &pDNNscArray[i];++i;//0;
   float *pDNNScore4900   = &pDNNscArray[i];++i;//0;
   float *pDNNScore5000   = &pDNNscArray[i];++i;//0;
   float *pDNNScore5100   = &pDNNscArray[i];++i;//0;
   float *pDNNScore5200   = &pDNNscArray[i];++i;//0;
   float *pDNNScore5300   = &pDNNscArray[i];++i;//0;
   float *pDNNScore5400   = &pDNNscArray[i];++i;//0;
   float *pDNNScore5500   = &pDNNscArray[i];++i;//0;
   float *pDNNScore5600   = &pDNNscArray[i];++i;//0;
   float *pDNNScore5700   = &pDNNscArray[i];++i;//0;
   float *pDNNScore5800   = &pDNNscArray[i];++i;//0;
   float *pDNNScore5900   = &pDNNscArray[i];++i;//0;
   float *pDNNScore6000   = &pDNNscArray[i];++i;//0;
   float *pDNNScore6100   = &pDNNscArray[i];++i;//0;
   float *pDNNScore6200   = &pDNNscArray[i];++i;//0;
   float *pDNNScore6300   = &pDNNscArray[i];++i;//0;
   float *pDNNScore6400   = &pDNNscArray[i];++i;//0;
   float *pDNNScore6500   = &pDNNscArray[i];++i;//0;
   unsigned int nWords = i;	     
   std::cout<<"Number of variables = "<<nWords<<std::endl;
			    


   
   
   
   /*
   TBranch *bptpDNNScore500    = newtree->Branch("pDNNScore500"   ,&pDNNScore500            ,"pDNNScore500/F");  
   TBranch *bptpDNNScore600    = newtree->Branch("pDNNScore600"   ,&pDNNScore600	    ,"pDNNScore600/F");  
   TBranch *bptpDNNScore700    = newtree->Branch("pDNNScore700"   ,&pDNNScore700	    ,"pDNNScore700/F"); 
   TBranch *bptpDNNScore800    = newtree->Branch("pDNNScore800"   ,&pDNNScore800	    ,"pDNNScore800/F");  
   TBranch *bptpDNNScore900    = newtree->Branch("pDNNScore900"   ,&pDNNScore900	    ,"pDNNScore900/F");    
   TBranch *bptpDNNScore1000   = newtree->Branch("pDNNScore1000"  ,&pDNNScore1000	    ,"pDNNScore1000/F"); 
   TBranch *bptpDNNScore1100   = newtree->Branch("pDNNScore1100"  ,&pDNNScore1100	    ,"pDNNScore1100/F"); 
   TBranch *bptpDNNScore1200   = newtree->Branch("pDNNScore1200"  ,&pDNNScore1200	    ,"pDNNScore1200/F"); 
   TBranch *bptpDNNScore1300   = newtree->Branch("pDNNScore1300"  ,&pDNNScore1300	    ,"pDNNScore1300/F"); 
   TBranch *bptpDNNScore1400   = newtree->Branch("pDNNScore1400"  ,&pDNNScore1400	    ,"pDNNScore1400/F"); 
   TBranch *bptpDNNScore1500   = newtree->Branch("pDNNScore1500"  ,&pDNNScore1500	    ,"pDNNScore1500/F"); 
   TBranch *bptpDNNScore1600   = newtree->Branch("pDNNScore1600"  ,&pDNNScore1600	    ,"pDNNScore1600/F"); 
   TBranch *bptpDNNScore1700   = newtree->Branch("pDNNScore1700"  ,&pDNNScore1700	    ,"pDNNScore1700/F"); 
   TBranch *bptpDNNScore1800   = newtree->Branch("pDNNScore1800"  ,&pDNNScore1800	    ,"pDNNScore1800/F"); 
   TBranch *bptpDNNScore1900   = newtree->Branch("pDNNScore1900"  ,&pDNNScore1900	    ,"pDNNScore1900/F"); 
   TBranch *bptpDNNScore2000   = newtree->Branch("pDNNScore2000"  ,&pDNNScore2000	    ,"pDNNScore2000/F");
   TBranch *bptpDNNScore2100   = newtree->Branch("pDNNScore2100"  ,&pDNNScore2100	    ,"pDNNScore2100/F");
   TBranch *bptpDNNScore2200   = newtree->Branch("pDNNScore2200"  ,&pDNNScore2200	    ,"pDNNScore2200/F");
   TBranch *bptpDNNScore2300   = newtree->Branch("pDNNScore2300"  ,&pDNNScore2300	    ,"pDNNScore2300/F");
   TBranch *bptpDNNScore2400   = newtree->Branch("pDNNScore2400"  ,&pDNNScore2400	    ,"pDNNScore2400/F");
   TBranch *bptpDNNScore2500   = newtree->Branch("pDNNScore2500"  ,&pDNNScore2500	    ,"pDNNScore2500/F");
   TBranch *bptpDNNScore2600   = newtree->Branch("pDNNScore2600"  ,&pDNNScore2600	    ,"pDNNScore2600/F");
   TBranch *bptpDNNScore2700   = newtree->Branch("pDNNScore2700"  ,&pDNNScore2700	    ,"pDNNScore2700/F");
   TBranch *bptpDNNScore2800   = newtree->Branch("pDNNScore2800"  ,&pDNNScore2800	    ,"pDNNScore2800/F");
   TBranch *bptpDNNScore2900   = newtree->Branch("pDNNScore2900"  ,&pDNNScore2900	    ,"pDNNScore2900/F");
   TBranch *bptpDNNScore3000   = newtree->Branch("pDNNScore3000"  ,&pDNNScore3000	    ,"pDNNScore3000/F");
   TBranch *bptpDNNScore3100   = newtree->Branch("pDNNScore3100"  ,&pDNNScore3100	    ,"pDNNScore3100/F");
   TBranch *bptpDNNScore3200   = newtree->Branch("pDNNScore3200"  ,&pDNNScore3200	    ,"pDNNScore3200/F");
   TBranch *bptpDNNScore3300   = newtree->Branch("pDNNScore3300"  ,&pDNNScore3300	    ,"pDNNScore3300/F");
   TBranch *bptpDNNScore3400   = newtree->Branch("pDNNScore3400"  ,&pDNNScore3400	    ,"pDNNScore3400/F");
   TBranch *bptpDNNScore3500   = newtree->Branch("pDNNScore3500"  ,&pDNNScore3500	    ,"pDNNScore3500/F");
   TBranch *bptpDNNScore3600   = newtree->Branch("pDNNScore3600"  ,&pDNNScore3600	    ,"pDNNScore3600/F");
   TBranch *bptpDNNScore3700   = newtree->Branch("pDNNScore3700"  ,&pDNNScore3700	    ,"pDNNScore3700/F");
   TBranch *bptpDNNScore3800   = newtree->Branch("pDNNScore3800"  ,&pDNNScore3800	    ,"pDNNScore3800/F");
   TBranch *bptpDNNScore3900   = newtree->Branch("pDNNScore3900"  ,&pDNNScore3900	    ,"pDNNScore3900/F");
   TBranch *bptpDNNScore4000   = newtree->Branch("pDNNScore4000"  ,&pDNNScore4000	    ,"pDNNScore4000/F");
   TBranch *bptpDNNScore4100   = newtree->Branch("pDNNScore4100"  ,&pDNNScore4100	    ,"pDNNScore4100/F");
   TBranch *bptpDNNScore4200   = newtree->Branch("pDNNScore4200"  ,&pDNNScore4200	    ,"pDNNScore4200/F");
   TBranch *bptpDNNScore4300   = newtree->Branch("pDNNScore4300"  ,&pDNNScore4300	    ,"pDNNScore4300/F");
   TBranch *bptpDNNScore4400   = newtree->Branch("pDNNScore4400"  ,&pDNNScore4400	    ,"pDNNScore4400/F");
   TBranch *bptpDNNScore4500   = newtree->Branch("pDNNScore4500"  ,&pDNNScore4500	    ,"pDNNScore4500/F");
   TBranch *bptpDNNScore4600   = newtree->Branch("pDNNScore4600"  ,&pDNNScore4600	    ,"pDNNScore4600/F");
   TBranch *bptpDNNScore4700   = newtree->Branch("pDNNScore4700"  ,&pDNNScore4700	    ,"pDNNScore4700/F");
   TBranch *bptpDNNScore4800   = newtree->Branch("pDNNScore4800"  ,&pDNNScore4800	    ,"pDNNScore4800/F");
   TBranch *bptpDNNScore4900   = newtree->Branch("pDNNScore4900"  ,&pDNNScore4900	    ,"pDNNScore4900/F");
   TBranch *bptpDNNScore5000   = newtree->Branch("pDNNScore5000"  ,&pDNNScore5000	    ,"pDNNScore5000/F");
   TBranch *bptpDNNScore5100   = newtree->Branch("pDNNScore5100"  ,&pDNNScore5100	    ,"pDNNScore5100/F");
   TBranch *bptpDNNScore5200   = newtree->Branch("pDNNScore5200"  ,&pDNNScore5200	    ,"pDNNScore5200/F");
   TBranch *bptpDNNScore5300   = newtree->Branch("pDNNScore5300"  ,&pDNNScore5300	    ,"pDNNScore5300/F");
   TBranch *bptpDNNScore5400   = newtree->Branch("pDNNScore5400"  ,&pDNNScore5400	    ,"pDNNScore5400/F");
   TBranch *bptpDNNScore5500   = newtree->Branch("pDNNScore5500"  ,&pDNNScore5500	    ,"pDNNScore5500/F");
   TBranch *bptpDNNScore5600   = newtree->Branch("pDNNScore5600"  ,&pDNNScore5600	    ,"pDNNScore5600/F");
   TBranch *bptpDNNScore5700   = newtree->Branch("pDNNScore5700"  ,&pDNNScore5700	    ,"pDNNScore5700/F");
   TBranch *bptpDNNScore5800   = newtree->Branch("pDNNScore5800"  ,&pDNNScore5800	    ,"pDNNScore5800/F");
   TBranch *bptpDNNScore5900   = newtree->Branch("pDNNScore5900"  ,&pDNNScore5900	    ,"pDNNScore5900/F");
   TBranch *bptpDNNScore6000   = newtree->Branch("pDNNScore6000"  ,&pDNNScore6000	    ,"pDNNScore6000/F");
   TBranch *bptpDNNScore6100   = newtree->Branch("pDNNScore6100"  ,&pDNNScore6100	    ,"pDNNScore6100/F");
   TBranch *bptpDNNScore6200   = newtree->Branch("pDNNScore6200"  ,&pDNNScore6200	    ,"pDNNScore6200/F");
   TBranch *bptpDNNScore6300   = newtree->Branch("pDNNScore6300"  ,&pDNNScore6300	    ,"pDNNScore6300/F");
   TBranch *bptpDNNScore6400   = newtree->Branch("pDNNScore6400"  ,&pDNNScore6400	    ,"pDNNScore6400/F");
   TBranch *bptpDNNScore6500   = newtree->Branch("pDNNScore6500"  ,&pDNNScore6500	    ,"pDNNScore6500/F");
   */
   i = 0;
   TBranch *bptpDNNScore[61]= {NULL};
   bptpDNNScore[i]   = newtree->Branch("pDNNScore500"   ,pDNNScore500       ,"pDNNScore500/F" );++i;  
   bptpDNNScore[i]   = newtree->Branch("pDNNScore600"   ,pDNNScore600	    ,"pDNNScore600/F" );++i;  
   bptpDNNScore[i]   = newtree->Branch("pDNNScore700"   ,pDNNScore700	    ,"pDNNScore700/F" );++i; 
   bptpDNNScore[i]   = newtree->Branch("pDNNScore800"   ,pDNNScore800	    ,"pDNNScore800/F" );++i;  
   bptpDNNScore[i]   = newtree->Branch("pDNNScore900"   ,pDNNScore900	    ,"pDNNScore900/F" );++i;    
   bptpDNNScore[i]   = newtree->Branch("pDNNScore1000"  ,pDNNScore1000	    ,"pDNNScore1000/F");++i; 
   bptpDNNScore[i]   = newtree->Branch("pDNNScore1100"  ,pDNNScore1100	    ,"pDNNScore1100/F");++i; 
   bptpDNNScore[i]   = newtree->Branch("pDNNScore1200"  ,pDNNScore1200	    ,"pDNNScore1200/F");++i; 
   bptpDNNScore[i]   = newtree->Branch("pDNNScore1300"  ,pDNNScore1300	    ,"pDNNScore1300/F");++i; 
   bptpDNNScore[i]   = newtree->Branch("pDNNScore1400"  ,pDNNScore1400	    ,"pDNNScore1400/F");++i; 
   bptpDNNScore[i]   = newtree->Branch("pDNNScore1500"  ,pDNNScore1500	    ,"pDNNScore1500/F");++i; 
   bptpDNNScore[i]   = newtree->Branch("pDNNScore1600"  ,pDNNScore1600	    ,"pDNNScore1600/F");++i; 
   bptpDNNScore[i]   = newtree->Branch("pDNNScore1700"  ,pDNNScore1700	    ,"pDNNScore1700/F");++i; 
   bptpDNNScore[i]   = newtree->Branch("pDNNScore1800"  ,pDNNScore1800	    ,"pDNNScore1800/F");++i; 
   bptpDNNScore[i]   = newtree->Branch("pDNNScore1900"  ,pDNNScore1900	    ,"pDNNScore1900/F");++i; 
   bptpDNNScore[i]   = newtree->Branch("pDNNScore2000"  ,pDNNScore2000	    ,"pDNNScore2000/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore2100"  ,pDNNScore2100	    ,"pDNNScore2100/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore2200"  ,pDNNScore2200	    ,"pDNNScore2200/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore2300"  ,pDNNScore2300	    ,"pDNNScore2300/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore2400"  ,pDNNScore2400	    ,"pDNNScore2400/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore2500"  ,pDNNScore2500	    ,"pDNNScore2500/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore2600"  ,pDNNScore2600	    ,"pDNNScore2600/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore2700"  ,pDNNScore2700	    ,"pDNNScore2700/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore2800"  ,pDNNScore2800	    ,"pDNNScore2800/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore2900"  ,pDNNScore2900	    ,"pDNNScore2900/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore3000"  ,pDNNScore3000	    ,"pDNNScore3000/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore3100"  ,pDNNScore3100	    ,"pDNNScore3100/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore3200"  ,pDNNScore3200	    ,"pDNNScore3200/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore3300"  ,pDNNScore3300	    ,"pDNNScore3300/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore3400"  ,pDNNScore3400	    ,"pDNNScore3400/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore3500"  ,pDNNScore3500	    ,"pDNNScore3500/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore3600"  ,pDNNScore3600	    ,"pDNNScore3600/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore3700"  ,pDNNScore3700	    ,"pDNNScore3700/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore3800"  ,pDNNScore3800	    ,"pDNNScore3800/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore3900"  ,pDNNScore3900	    ,"pDNNScore3900/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore4000"  ,pDNNScore4000	    ,"pDNNScore4000/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore4100"  ,pDNNScore4100	    ,"pDNNScore4100/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore4200"  ,pDNNScore4200	    ,"pDNNScore4200/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore4300"  ,pDNNScore4300	    ,"pDNNScore4300/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore4400"  ,pDNNScore4400	    ,"pDNNScore4400/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore4500"  ,pDNNScore4500	    ,"pDNNScore4500/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore4600"  ,pDNNScore4600	    ,"pDNNScore4600/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore4700"  ,pDNNScore4700	    ,"pDNNScore4700/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore4800"  ,pDNNScore4800	    ,"pDNNScore4800/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore4900"  ,pDNNScore4900	    ,"pDNNScore4900/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore5000"  ,pDNNScore5000	    ,"pDNNScore5000/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore5100"  ,pDNNScore5100	    ,"pDNNScore5100/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore5200"  ,pDNNScore5200	    ,"pDNNScore5200/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore5300"  ,pDNNScore5300	    ,"pDNNScore5300/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore5400"  ,pDNNScore5400	    ,"pDNNScore5400/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore5500"  ,pDNNScore5500	    ,"pDNNScore5500/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore5600"  ,pDNNScore5600	    ,"pDNNScore5600/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore5700"  ,pDNNScore5700	    ,"pDNNScore5700/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore5800"  ,pDNNScore5800	    ,"pDNNScore5800/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore5900"  ,pDNNScore5900	    ,"pDNNScore5900/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore6000"  ,pDNNScore6000	    ,"pDNNScore6000/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore6100"  ,pDNNScore6100	    ,"pDNNScore6100/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore6200"  ,pDNNScore6200	    ,"pDNNScore6200/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore6300"  ,pDNNScore6300	    ,"pDNNScore6300/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore6400"  ,pDNNScore6400	    ,"pDNNScore6400/F");++i;
   bptpDNNScore[i]   = newtree->Branch("pDNNScore6500"  ,pDNNScore6500	    ,"pDNNScore6500/F");++i;
   std::cout<<"All branches added to the new tree "<<std::endl;



   Bool_t Pass_MergHP_GGF_ZZ_Tag_SR        ;
   Bool_t Pass_MergHP_GGF_ZZ_Untag_SR      ;
   Bool_t Pass_MergLP_GGF_ZZ_Tag_SR        ;
   Bool_t Pass_MergLP_GGF_ZZ_Untag_SR      ;
   Bool_t Pass_Res_GGF_ZZ_Tag_SR           ;
   Bool_t Pass_Res_GGF_ZZ_Untag_SR         ;
   	                                  
   Bool_t Pass_MergHP_VBF_ZZ_SR            ;
   Bool_t Pass_MergLP_VBF_ZZ_SR            ;
   Bool_t Pass_Res_VBF_ZZ_SR               ;
	                                  
   Bool_t Pass_MergHP_GGF_WZ_SR            ;
   Bool_t Pass_MergLP_GGF_WZ_SR            ;
   Bool_t Pass_Res_GGF_WZ_SR               ;
   	                                  
   Bool_t Pass_MergHP_VBF_WZ_SR            ;
   Bool_t Pass_MergLP_VBF_WZ_SR            ;
   Bool_t Pass_Res_VBF_WZ_SR               ;

   Bool_t Pass_MergHP_GGF_ZZ_Tag_ZCR       ;
   Bool_t Pass_MergHP_GGF_ZZ_Untag_ZCR     ;
   Bool_t Pass_MergLP_GGF_ZZ_Tag_ZCR       ;
   Bool_t Pass_MergLP_GGF_ZZ_Untag_ZCR     ;
   Bool_t Pass_Res_GGF_ZZ_Tag_ZCR          ;
   Bool_t Pass_Res_GGF_ZZ_Untag_ZCR        ;
   
   Bool_t Pass_MergHP_VBF_ZZ_ZCR           ;
   Bool_t Pass_MergLP_VBF_ZZ_ZCR           ;
   Bool_t Pass_Res_VBF_ZZ_ZCR              ;

   Bool_t Pass_MergHP_GGF_WZ_ZCR           ;
   Bool_t Pass_MergLP_GGF_WZ_ZCR           ;
   Bool_t Pass_Res_GGF_WZ_ZCR              ;
   
   Bool_t Pass_MergHP_VBF_WZ_ZCR           ;
   Bool_t Pass_MergLP_VBF_WZ_ZCR           ;
   Bool_t Pass_Res_VBF_WZ_ZCR              ;


   newtree->SetBranchAddress("Pass_MergHP_GGF_ZZ_Tag_SR"     ,&Pass_MergHP_GGF_ZZ_Tag_SR    );
   newtree->SetBranchAddress("Pass_MergHP_GGF_ZZ_Untag_SR"   ,&Pass_MergHP_GGF_ZZ_Untag_SR  );
   newtree->SetBranchAddress("Pass_MergLP_GGF_ZZ_Tag_SR"     ,&Pass_MergLP_GGF_ZZ_Tag_SR    );
   newtree->SetBranchAddress("Pass_MergLP_GGF_ZZ_Untag_SR"   ,&Pass_MergLP_GGF_ZZ_Untag_SR  );
   newtree->SetBranchAddress("Pass_Res_GGF_ZZ_Tag_SR"        ,&Pass_Res_GGF_ZZ_Tag_SR       );
   newtree->SetBranchAddress("Pass_Res_GGF_ZZ_Untag_SR"      ,&Pass_Res_GGF_ZZ_Untag_SR     );
   newtree->SetBranchAddress("Pass_MergHP_VBF_ZZ_SR"         ,&Pass_MergHP_VBF_ZZ_SR        );
   newtree->SetBranchAddress("Pass_MergLP_VBF_ZZ_SR"         ,&Pass_MergLP_VBF_ZZ_SR        );
   newtree->SetBranchAddress("Pass_Res_VBF_ZZ_SR"            ,&Pass_Res_VBF_ZZ_SR           );
   newtree->SetBranchAddress("Pass_MergHP_GGF_WZ_SR"         ,&Pass_MergHP_GGF_WZ_SR        );
   newtree->SetBranchAddress("Pass_MergLP_GGF_WZ_SR"         ,&Pass_MergLP_GGF_WZ_SR        );
   newtree->SetBranchAddress("Pass_Res_GGF_WZ_SR"            ,&Pass_Res_GGF_WZ_SR           );
   newtree->SetBranchAddress("Pass_MergHP_VBF_WZ_SR"         ,&Pass_MergHP_VBF_WZ_SR        );
   newtree->SetBranchAddress("Pass_MergLP_VBF_WZ_SR"         ,&Pass_MergLP_VBF_WZ_SR        );
   newtree->SetBranchAddress("Pass_Res_VBF_WZ_SR"            ,&Pass_Res_VBF_WZ_SR           );      

   newtree->SetBranchAddress("Pass_MergHP_GGF_ZZ_Tag_ZCR"     ,&Pass_MergHP_GGF_ZZ_Tag_ZCR    );
   newtree->SetBranchAddress("Pass_MergHP_GGF_ZZ_Untag_ZCR"   ,&Pass_MergHP_GGF_ZZ_Untag_ZCR  );
   newtree->SetBranchAddress("Pass_MergLP_GGF_ZZ_Tag_ZCR"     ,&Pass_MergLP_GGF_ZZ_Tag_ZCR    );
   newtree->SetBranchAddress("Pass_MergLP_GGF_ZZ_Untag_ZCR"   ,&Pass_MergLP_GGF_ZZ_Untag_ZCR  );
   newtree->SetBranchAddress("Pass_Res_GGF_ZZ_Tag_ZCR"        ,&Pass_Res_GGF_ZZ_Tag_ZCR       );
   newtree->SetBranchAddress("Pass_Res_GGF_ZZ_Untag_ZCR"      ,&Pass_Res_GGF_ZZ_Untag_ZCR     );
   newtree->SetBranchAddress("Pass_MergHP_VBF_ZZ_ZCR"         ,&Pass_MergHP_VBF_ZZ_ZCR        );
   newtree->SetBranchAddress("Pass_MergLP_VBF_ZZ_ZCR"         ,&Pass_MergLP_VBF_ZZ_ZCR        );
   newtree->SetBranchAddress("Pass_Res_VBF_ZZ_ZCR"            ,&Pass_Res_VBF_ZZ_ZCR           );
   newtree->SetBranchAddress("Pass_MergHP_GGF_WZ_ZCR"         ,&Pass_MergHP_GGF_WZ_ZCR        );
   newtree->SetBranchAddress("Pass_MergLP_GGF_WZ_ZCR"         ,&Pass_MergLP_GGF_WZ_ZCR        );
   newtree->SetBranchAddress("Pass_Res_GGF_WZ_ZCR"            ,&Pass_Res_GGF_WZ_ZCR           );
   newtree->SetBranchAddress("Pass_MergHP_VBF_WZ_ZCR"         ,&Pass_MergHP_VBF_WZ_ZCR        );
   newtree->SetBranchAddress("Pass_MergLP_VBF_WZ_ZCR"         ,&Pass_MergLP_VBF_WZ_ZCR        );
   newtree->SetBranchAddress("Pass_Res_VBF_WZ_ZCR"            ,&Pass_Res_VBF_WZ_ZCR           );      
   std::cout<<"Address for branches of flags found "<<std::endl;


   

   bool sel_Merg_ggF = Pass_MergHP_GGF_ZZ_Tag_SR || Pass_MergHP_GGF_ZZ_Untag_SR  || Pass_MergHP_GGF_WZ_SR  || Pass_MergLP_GGF_ZZ_Tag_SR  || Pass_MergLP_GGF_ZZ_Untag_SR  || Pass_MergLP_GGF_WZ_SR  || Pass_MergHP_GGF_ZZ_Tag_ZCR  || Pass_MergHP_GGF_ZZ_Untag_ZCR  || Pass_MergHP_GGF_WZ_ZCR  || Pass_MergLP_GGF_ZZ_Tag_ZCR  || Pass_MergLP_GGF_ZZ_Untag_ZCR  || Pass_MergLP_GGF_WZ_ZCR ;
   bool sel_Merg_VBF = Pass_MergHP_VBF_WZ_SR ||Pass_MergHP_VBF_ZZ_SR ||Pass_MergLP_VBF_WZ_SR ||Pass_MergLP_VBF_ZZ_SR ||Pass_MergHP_VBF_WZ_ZCR ||Pass_MergHP_VBF_ZZ_ZCR ||Pass_MergLP_VBF_WZ_ZCR ||Pass_MergLP_VBF_ZZ_ZCR ;
   bool sel_Res_ggF = Pass_Res_GGF_WZ_SR ||Pass_Res_GGF_ZZ_Tag_SR ||Pass_Res_GGF_ZZ_Untag_SR ||Pass_Res_GGF_WZ_ZCR ||Pass_Res_GGF_ZZ_Tag_ZCR ||Pass_Res_GGF_ZZ_Untag_ZCR ;
   bool sel_Res_VBF = Pass_Res_VBF_WZ_SR  || Pass_Res_VBF_ZZ_SR  || Pass_Res_VBF_WZ_ZCR  || Pass_Res_VBF_ZZ_ZCR ;
   
   bool selection=false;
   if (prodChannel == "all")
     {
       selection = sel_Res_VBF || sel_Merg_VBF || sel_Res_ggF || sel_Merg_ggF; 
     }
   else if (prodChannel=="ggF")
     {
       if (analysis == "merged")        selection =  sel_Merg_ggF;
       else if (analysis == "resolved") selection =  sel_Res_ggF;
     }
   else if (prodChannel=="VBF")
     {
       if (analysis == "merged")        selection =  sel_Merg_VBF;
       else if (analysis == "resolved") selection =  sel_Res_VBF;
     }
   std::cout<<" Selection defined"<<std::endl;
   std::ifstream scoreInputFile;
   //std::string fn = "/afs/le.infn.it/user/s/spagnolo/html/allow_listing/DBL/tmp/yield_"+tag+"_"+myvar+".txt";
   //std::string fn = "/afs/le.infn.it/user/s/spagnolo/html/allow_listing/DBL/tmp/yield_Run2_AllowOverlap_"+myvar+".txt";
   // std::string fn = inputScoreFileN;// "/afs/le.infn.it/user/s/spagnolo/html/allow_listing/DBL/tmp/yield_Run2_AllowOverlap_ZZandWZ.txt";
   std::cout<<" Reading scores from input file "<<fn<<std::endl;
   scoreInputFile.open(fn.c_str(), std::ios_base::in);
   stringstream ssOneLine;
   string sOneLine;
   int nLines = 0;
   if (scoreInputFile.is_open()) {
     for(int i=0; i<newtree->GetEntries(); i++){
       newtree->GetEntry(i);
       
       bool sel_Merg_ggF = Pass_MergHP_GGF_ZZ_Tag_SR || Pass_MergHP_GGF_ZZ_Untag_SR  || Pass_MergHP_GGF_WZ_SR  || Pass_MergLP_GGF_ZZ_Tag_SR  || Pass_MergLP_GGF_ZZ_Untag_SR  || Pass_MergLP_GGF_WZ_SR  || Pass_MergHP_GGF_ZZ_Tag_ZCR  || Pass_MergHP_GGF_ZZ_Untag_ZCR  || Pass_MergHP_GGF_WZ_ZCR  || Pass_MergLP_GGF_ZZ_Tag_ZCR  || Pass_MergLP_GGF_ZZ_Untag_ZCR  || Pass_MergLP_GGF_WZ_ZCR ;
       bool sel_Merg_VBF = Pass_MergHP_VBF_WZ_SR ||Pass_MergHP_VBF_ZZ_SR ||Pass_MergLP_VBF_WZ_SR ||Pass_MergLP_VBF_ZZ_SR ||Pass_MergHP_VBF_WZ_ZCR ||Pass_MergHP_VBF_ZZ_ZCR ||Pass_MergLP_VBF_WZ_ZCR ||Pass_MergLP_VBF_ZZ_ZCR ;
       bool sel_Res_ggF = Pass_Res_GGF_WZ_SR ||Pass_Res_GGF_ZZ_Tag_SR ||Pass_Res_GGF_ZZ_Untag_SR ||Pass_Res_GGF_WZ_ZCR ||Pass_Res_GGF_ZZ_Tag_ZCR ||Pass_Res_GGF_ZZ_Untag_ZCR ;
       bool sel_Res_VBF = Pass_Res_VBF_WZ_SR  || Pass_Res_VBF_ZZ_SR  || Pass_Res_VBF_WZ_ZCR  || Pass_Res_VBF_ZZ_ZCR ;
       
       bool selection=false;
       if (prodChannel == "all")
	 {
	   selection = sel_Res_VBF || sel_Merg_VBF || sel_Res_ggF || sel_Merg_ggF; 
	 }
       else if (prodChannel=="ggF")
	 {
	   if (analysis == "merged")        selection =  sel_Merg_ggF;
	   else if (analysis == "resolved") selection =  sel_Res_ggF;
	 }
       else if (prodChannel=="VBF")
	 {
	   if (analysis == "merged")        selection =  sel_Merg_VBF;
	   else if (analysis == "resolved") selection =  sel_Res_VBF;
	 }
       //std::cout<<" Selection defined"<<std::endl;
       
       if (selection)
	 {
	   std::getline(scoreInputFile, sOneLine);
	   ssOneLine <<sOneLine;
	   ++nLines;
	   if (nLines%10000==0 || nLines==1) {std::cout <<nLines<<" lines read so far ***************************"<< std::endl;
	     cout<<sOneLine<<std::endl;
	   }
	   for (unsigned int iw=0; iw<nWords; ++iw) {
	     ssOneLine>>pDNNscArray[iw];
	     bptpDNNScore[iw]->Fill();
	     //if (fabs(pDNNscArray[iw])>0.01 && fabs(pDNNscArray[iw]-1.)>0.01) std::cout<<" iw, fabs(pDNNscArray[iw] "<<iw<<" "<<pDNNscArray[iw]<<" score600: "<<*pDNNScore600<<std::endl;
	     //if (fabs(pDNNscArray[iw]-1.)<0.01) std::cout<<" iw, fabs(pDNNscArray[iw] "<<iw<<" "<<pDNNscArray[iw]<<std::endl;
	   }
	   if (nLines%10000==0) std::cout <<nLines<<" pDNNScore500, pDNNScore3000, pDNNScore6500 = "<<(*pDNNScore500)<<" "<<(*pDNNScore3000)<<" "<<(*pDNNScore6500)<<std::endl;
	 }
       else
	 {
	   //continue;
	   for (unsigned int iw=0; iw<nWords; ++iw) {
	     pDNNscArray[iw]=-1.;
	     bptpDNNScore[iw]->Fill();
	   }
	 }
     }
     std::cout<<"Number of lines read = "<<nLines<<std::endl;
   }
   else {
     // show message:
     std::cout << "Error opening score input file: <"<<fn<<">"<<std::endl;
     return;
   }

   /*
   for (int i = 0; i<newtree->GetEntries(); ++i)
     {
       pDNNScore500 = (float)i;
       bptpDNNScore500->Fill();
       pDNNScore1000 = (float)(i+100);
       bptpDNNScore1000->Fill();
     }
   */

   
   TCanvas* c = new TCanvas("c","c",800,600);
   c->SetLogy();
   TH1F* h1 = new TH1F("h","h",102,-0.01,1.01);
   newtree->Draw("pDNNScore600>>h","");
   std::cout<<"pDNNScore600: entries, mean, rms = "<<h1->GetEntries()<<" "<<h1->GetMean()<<" "<<h1->GetRMS()<<std::endl;
   c->SaveAs("provaRoot.pdf");
  

   
   //newtree->Print();
   newfile.Write();
   std::cout<<"New tree written in file: "<<newfile.GetName()<<std::endl;
   return;
}
