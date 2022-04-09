//#include "chains/Chains_33_22_pdnn.h"
#include "chains/chainSet_33_22_pdnn.h"
#include "chains/readChains_33_22_pdnnScores.C"


#include <string>


// #include "TChain.h"
// TChain* _fzjet;
// TChain* _fDB  ; 
// TChain* _fwjet  ;
// TChain* _fttbar ; 
// TChain* _fstop  ; 
// TChain* _fdata; 
// TChain* _fsignal; 

// /*
// TChain* _fRadion; 
// TChain* _fHVT; 
// TChain* _fRSG; 
// TChain* _fVBFRadion; 
// TChain* _fVBFHVT; 
// TChain* _fVBFRSG; 
// */

// TChain* _fadata[3][2][2]; 
// TChain* _fazjet[3][2][2];
// TChain* _faDB[3][2][2]  ; 
// TChain* _fawjet[3][2][2]  ;
// TChain* _fattbar[3][2][2] ; 
// TChain* _fastop[3][2][2]  ; 

// /*
// TChain* _fRadion[3][2][2]; 
// TChain* _fHVT[3][2][2]; 
// TChain* _fRSG[3][2][2]; 
// TChain* _fVBFRadion[3][2][2]; 
// TChain* _fVBFHVT[3][2][2]; 
// TChain* _fVBFRSG[3][2][2];
// */

// TChain* _faSignal[3][2][2];

void initChains_33_22_pdnn(std::string scoreSignal="RSG", std::string scoreAnalysis="merged", std::string scoreChannel="ggF", chainSet_33_22_pdnn* cs=NULL)
{

  std::cout<<"Initialize chains"<<std::endl;
  if (cs==NULL) {std::cout<<"initChains_33_22_pdnn::NULL pointer to chains-set"<<std::endl;return;}
  int is = -1;
  int ic = -1;
  int ia = -1;
  

  if        (scoreSignal=="RSG")      is = 0;
  else if   (scoreSignal=="Radion")   is = 1;
  else if   (scoreSignal=="HVTWZ")      is = 2;
  else      std::cout<<"Error scoreSignal must be <RSG>, <Radion> or <HVTWZ>, now = "<<scoreSignal<<std::endl;

  if        (scoreChannel=="ggF")     ic = 0;
  else if   (scoreChannel=="VBF")     ic = 1;
  else      std::cout<<"Error scoreChannel must be <ggF> or <VBF>,            now = "<<scoreChannel<<std::endl;

  if        (scoreAnalysis=="merged")  ia = 0;
  else if   (scoreAnalysis=="resolved")ia = 1;
  else      std::cout<<"Error scoreAnalysis must be <merged> or <resolved>,   now = "<<scoreAnalysis<<std::endl;

  if (is < 0 || ic < 0 || ia < 0 ) return;

  
  if ( cs->_fazjet[is][ic][ia] == NULL )
    {
      std::cout<<"Create and read from file ... signal/analysis/channel "<<scoreSignal<<"/"<<scoreAnalysis<<"/"<<scoreChannel<<std::endl;
          cs->_fazjet [is][ic][ia]    = new TChain("Nominal");;
	  cs->_fawjet [is][ic][ia]    = new TChain("Nominal");;
	  cs->_faDB   [is][ic][ia]    = new TChain("Nominal");; 
	  cs->_fattbar[is][ic][ia]    = new TChain("Nominal");;
	  cs->_fastop [is][ic][ia]    = new TChain("Nominal");;
	  
	  cs->_fadata [is][ic][ia]    = new TChain("Nominal"); 
	  /*
	  cs->_faRadion    [is][ic][ia]= new TChain("Nominal"); 
	  cs->_faHVT       [is][ic][ia]= new TChain("Nominal");  
	  cs->_faRSG       [is][ic][ia]= new TChain("Nominal"); 
	  
	  cs->_faVBFRadion [is][ic][ia]= new TChain("Nominal"); 
	  cs->_faVBFHVT    [is][ic][ia]= new TChain("Nominal");  
	  cs->_faVBFRSG    [is][ic][ia]= new TChain("Nominal"); 
	  */
	  cs->_faSignal    [is][ic][ia]= new TChain("Nominal"); 
   
	  if        (scoreSignal=="RSG")
	    {
	      if        (scoreChannel=="ggF")
		{
		  if        (scoreAnalysis=="merged")
		    {
		      myChain33_22_pdnnForRSG_ggF_merged_Zjet   ( cs->_fazjet [is][ic][ia]        );
		      myChain33_22_pdnnForRSG_ggF_merged_Wjet   ( cs->_fawjet [is][ic][ia]        );
		      myChain33_22_pdnnForRSG_ggF_merged_Diboson( cs->_faDB   [is][ic][ia]        );
		      myChain33_22_pdnnForRSG_ggF_merged_ttbar  ( cs->_fattbar[is][ic][ia]        );
		      myChain33_22_pdnnForRSG_ggF_merged_stop   ( cs->_fastop [is][ic][ia]        );
		      myChain33_22_pdnnForRSG_ggF_merged_data   ( cs->_fadata [is][ic][ia]        );
	      							             
		      //myChain33_22_pdnnForRSG_ggF_merged_RSG    ( cs->_faRSG   [is][ic][ia]        );
		      //cs->_fsignal = cs->_faRSG[is][ic][ia]        ;
		      myChain33_22_pdnnForRSG_ggF_merged_RSG    ( cs->_faSignal [is][ic][ia]        );
		    }
		  else if   (scoreAnalysis=="resolved")
		    {
		      myChain33_22_pdnnForRSG_ggF_resolved_Zjet   ( cs->_fazjet [is][ic][ia]       );
		      myChain33_22_pdnnForRSG_ggF_resolved_Wjet   ( cs->_fawjet [is][ic][ia]       );
		      myChain33_22_pdnnForRSG_ggF_resolved_Diboson( cs->_faDB   [is][ic][ia]       );
		      myChain33_22_pdnnForRSG_ggF_resolved_ttbar  ( cs->_fattbar[is][ic][ia]       );
		      myChain33_22_pdnnForRSG_ggF_resolved_stop   ( cs->_fastop [is][ic][ia]       );
		      myChain33_22_pdnnForRSG_ggF_resolved_data   ( cs->_fadata [is][ic][ia]       );
	      							               
		      //myChain33_22_pdnnForRSG_ggF_resolved_RSG    ( cs->_faRSG   [is][ic][ia]       );
		      //cs->_fsignal = cs->_faRSG[is][ic][ia]        ;
		      myChain33_22_pdnnForRSG_ggF_resolved_RSG    ( cs->_faSignal   [is][ic][ia]       );
		    }
		}
	      else if   (scoreChannel=="VBF")
		{
		  if        (scoreAnalysis=="merged")
		    {
		      myChain33_22_pdnnForVBFRSG_VBF_merged_Zjet   ( cs->_fazjet   [is][ic][ia]     );
		      myChain33_22_pdnnForVBFRSG_VBF_merged_Wjet   ( cs->_fawjet   [is][ic][ia]     );
		      myChain33_22_pdnnForVBFRSG_VBF_merged_Diboson( cs->_faDB     [is][ic][ia]     );
		      myChain33_22_pdnnForVBFRSG_VBF_merged_ttbar  ( cs->_fattbar  [is][ic][ia]     );
		      myChain33_22_pdnnForVBFRSG_VBF_merged_stop   ( cs->_fastop   [is][ic][ia]     );
		      myChain33_22_pdnnForVBFRSG_VBF_merged_data   ( cs->_fadata   [is][ic][ia]     );
	      			            			                  
		      //myChain33_22_pdnnForVBFRSG_VBF_merged_VBFRSG ( cs->_faVBFRSG  [is][ic][ia]     );
		      //cs->_fsignal = cs->_faVBFRSG[is][ic][ia]        ;
		      myChain33_22_pdnnForVBFRSG_VBF_merged_VBFRSG ( cs->_faSignal  [is][ic][ia]     );
		    }
		  else if   (scoreAnalysis=="resolved")
		    {
		      myChain33_22_pdnnForVBFRSG_VBF_resolved_Zjet   ( cs->_fazjet  [is][ic][ia]      );
		      myChain33_22_pdnnForVBFRSG_VBF_resolved_Wjet   ( cs->_fawjet  [is][ic][ia]      );
		      myChain33_22_pdnnForVBFRSG_VBF_resolved_Diboson( cs->_faDB    [is][ic][ia]      );
		      myChain33_22_pdnnForVBFRSG_VBF_resolved_ttbar  ( cs->_fattbar [is][ic][ia]      );
		      myChain33_22_pdnnForVBFRSG_VBF_resolved_stop   ( cs->_fastop  [is][ic][ia]      );
		      myChain33_22_pdnnForVBFRSG_VBF_resolved_data   ( cs->_fadata  [is][ic][ia]      );
	      			     				                   
		      //myChain33_22_pdnnForVBFRSG_VBF_resolved_VBFRSG ( cs->_faVBFRSG [is][ic][ia]      );
		      //cs->_fsignal = cs->_faVBFRSG[is][ic][ia]        ;
		      myChain33_22_pdnnForVBFRSG_VBF_resolved_VBFRSG ( cs->_faSignal [is][ic][ia]      );
		    }
		}
	    }
	  else if   (scoreSignal=="Radion")
	    {
	      if        (scoreChannel=="ggF")
		{
		  if        (scoreAnalysis=="merged")
		    {
		      myChain33_22_pdnnForRadion_ggF_merged_Zjet   ( cs->_fazjet  [is][ic][ia]      );
		      myChain33_22_pdnnForRadion_ggF_merged_Wjet   ( cs->_fawjet  [is][ic][ia]      );
		      myChain33_22_pdnnForRadion_ggF_merged_Diboson( cs->_faDB    [is][ic][ia]      );
		      myChain33_22_pdnnForRadion_ggF_merged_ttbar  ( cs->_fattbar [is][ic][ia]      );
		      myChain33_22_pdnnForRadion_ggF_merged_stop   ( cs->_fastop  [is][ic][ia]      );
		      myChain33_22_pdnnForRadion_ggF_merged_data   ( cs->_fadata  [is][ic][ia]      );
	      							                 
		      //myChain33_22_pdnnForRadion_ggF_merged_Radion ( cs->_faRadion [is][ic][ia]      );
		      //cs->_fsignal = cs->_faRadion[is][ic][ia]        ;
		      myChain33_22_pdnnForRadion_ggF_merged_Radion ( cs->_faSignal [is][ic][ia]      );
		    }
		  else if   (scoreAnalysis=="resolved")
		    {
		      myChain33_22_pdnnForRadion_ggF_resolved_Zjet   ( cs->_fazjet  [is][ic][ia]      );
		      myChain33_22_pdnnForRadion_ggF_resolved_Wjet   ( cs->_fawjet  [is][ic][ia]      );
		      myChain33_22_pdnnForRadion_ggF_resolved_Diboson( cs->_faDB    [is][ic][ia]      );
		      myChain33_22_pdnnForRadion_ggF_resolved_ttbar  ( cs->_fattbar [is][ic][ia]      );
		      myChain33_22_pdnnForRadion_ggF_resolved_stop   ( cs->_fastop  [is][ic][ia]      );
		      myChain33_22_pdnnForRadion_ggF_resolved_data   ( cs->_fadata  [is][ic][ia]      );
	      							                   
		      //myChain33_22_pdnnForRadion_ggF_resolved_Radion ( cs->_faRadion [is][ic][ia]      );
		      //cs->_fsignal = cs->_faRadion[is][ic][ia]        ;
		      myChain33_22_pdnnForRadion_ggF_resolved_Radion ( cs->_faSignal [is][ic][ia]      );
		    }
		}
	      else if   (scoreChannel=="VBF")
		{
		  if        (scoreAnalysis=="merged")
		    {
		      myChain33_22_pdnnForVBFRadion_VBF_merged_Zjet   ( cs->_fazjet       [is][ic][ia] );
		      myChain33_22_pdnnForVBFRadion_VBF_merged_Wjet   ( cs->_fawjet       [is][ic][ia] );
		      myChain33_22_pdnnForVBFRadion_VBF_merged_Diboson( cs->_faDB         [is][ic][ia] );
		      myChain33_22_pdnnForVBFRadion_VBF_merged_ttbar  ( cs->_fattbar      [is][ic][ia] );
		      myChain33_22_pdnnForVBFRadion_VBF_merged_stop   ( cs->_fastop       [is][ic][ia] );
		      myChain33_22_pdnnForVBFRadion_VBF_merged_data   ( cs->_fadata       [is][ic][ia] );
	      			     					                 
		      //myChain33_22_pdnnForVBFRadion_VBF_merged_VBFRadion( cs->_faVBFRadion [is][ic][ia] );
		      //cs->_fsignal = cs->_faVBFRadion[is][ic][ia]        ;
		      myChain33_22_pdnnForVBFRadion_VBF_merged_VBFRadion( cs->_faSignal [is][ic][ia] );
		    }
		  else if   (scoreAnalysis=="resolved")
		    {
		      myChain33_22_pdnnForVBFRadion_VBF_resolved_Zjet   ( cs->_fazjet       [is][ic][ia] );
		      myChain33_22_pdnnForVBFRadion_VBF_resolved_Wjet   ( cs->_fawjet       [is][ic][ia] );
		      myChain33_22_pdnnForVBFRadion_VBF_resolved_Diboson( cs->_faDB         [is][ic][ia] );
		      myChain33_22_pdnnForVBFRadion_VBF_resolved_ttbar  ( cs->_fattbar      [is][ic][ia] );
		      myChain33_22_pdnnForVBFRadion_VBF_resolved_stop   ( cs->_fastop       [is][ic][ia] );
		      myChain33_22_pdnnForVBFRadion_VBF_resolved_data   ( cs->_fadata       [is][ic][ia] );
	      			     					                   
		      //myChain33_22_pdnnForVBFRadion_VBF_resolved_VBFRadion( cs->_faVBFRadion [is][ic][ia] );
		      //cs->_fsignal = cs->_faVBFRadion[is][ic][ia]        ;
		      myChain33_22_pdnnForVBFRadion_VBF_resolved_VBFRadion( cs->_faSignal [is][ic][ia] );
		    }
		}
	    }
	  else if   (scoreSignal=="HVTWZ")
	    {
	      if        (scoreChannel=="ggF")
		{
		  if        (scoreAnalysis=="merged")
		    {
		      myChain33_22_pdnnForHVTWZ_ggF_merged_Zjet   ( cs->_fazjet [is][ic][ia]       );
		      myChain33_22_pdnnForHVTWZ_ggF_merged_Wjet   ( cs->_fawjet [is][ic][ia]       );
		      myChain33_22_pdnnForHVTWZ_ggF_merged_Diboson( cs->_faDB   [is][ic][ia]       );
		      myChain33_22_pdnnForHVTWZ_ggF_merged_ttbar  ( cs->_fattbar[is][ic][ia]       );
		      myChain33_22_pdnnForHVTWZ_ggF_merged_stop   ( cs->_fastop [is][ic][ia]       );
		      myChain33_22_pdnnForHVTWZ_ggF_merged_data   ( cs->_fadata [is][ic][ia]       );
	      							               
		      //myChain33_22_pdnnForHVTWZ_ggF_merged_HVTWZ  ( cs->_faHVT   [is][ic][ia]       );
		      //cs->_fsignal = cs->_faHVT[is][ic][ia]        ;
		      myChain33_22_pdnnForHVTWZ_ggF_merged_HVTWZ  ( cs->_faSignal   [is][ic][ia]       );
		      cs->_fsignal = cs->_faSignal[is][ic][ia]        ;
		    }
		  else if   (scoreAnalysis=="resolved")
		    {
		      myChain33_22_pdnnForHVTWZ_ggF_resolved_Zjet   ( cs->_fazjet [is][ic][ia]       );
		      myChain33_22_pdnnForHVTWZ_ggF_resolved_Wjet   ( cs->_fawjet [is][ic][ia]       );
		      myChain33_22_pdnnForHVTWZ_ggF_resolved_Diboson( cs->_faDB   [is][ic][ia]       );
		      myChain33_22_pdnnForHVTWZ_ggF_resolved_ttbar  ( cs->_fattbar[is][ic][ia]       );
		      myChain33_22_pdnnForHVTWZ_ggF_resolved_stop   ( cs->_fastop [is][ic][ia]       );
		      myChain33_22_pdnnForHVTWZ_ggF_resolved_data   ( cs->_fadata [is][ic][ia]       );
	      							                 
		      //myChain33_22_pdnnForHVTWZ_ggF_resolved_HVTWZ  ( cs->_faHVT   [is][ic][ia]       );
		      //cs->_fsignal = cs->_faHVT[is][ic][ia]        ;
		      myChain33_22_pdnnForHVTWZ_ggF_resolved_HVTWZ  ( cs->_faSignal   [is][ic][ia]       );
		    }
		}
	      else if   (scoreChannel=="VBF")
		{
		  if        (scoreAnalysis=="merged")
		    {
		      myChain33_22_pdnnForVBFHVTWZ_VBF_merged_Zjet   ( cs->_fazjet   [is][ic][ia]     );
		      myChain33_22_pdnnForVBFHVTWZ_VBF_merged_Wjet   ( cs->_fawjet   [is][ic][ia]     );
		      myChain33_22_pdnnForVBFHVTWZ_VBF_merged_Diboson( cs->_faDB     [is][ic][ia]     );
		      myChain33_22_pdnnForVBFHVTWZ_VBF_merged_ttbar  ( cs->_fattbar  [is][ic][ia]     );
		      myChain33_22_pdnnForVBFHVTWZ_VBF_merged_stop   ( cs->_fastop   [is][ic][ia]     );
		      myChain33_22_pdnnForVBFHVTWZ_VBF_merged_data   ( cs->_fadata   [is][ic][ia]     );
	      			     					            
		      //myChain33_22_pdnnForVBFHVTWZ_VBF_merged_VBFHVTWZ( cs->_faVBFHVT [is][ic][ia]     );
		      //cs->_fsignal = cs->_faVBFHVT[is][ic][ia]        ;
		      myChain33_22_pdnnForVBFHVTWZ_VBF_merged_VBFHVTWZ( cs->_faSignal [is][ic][ia]     );
		    }
		  else if   (scoreAnalysis=="resolved")
		    {
		      myChain33_22_pdnnForVBFHVTWZ_VBF_resolved_Zjet   ( cs->_fazjet   [is][ic][ia]     );
		      myChain33_22_pdnnForVBFHVTWZ_VBF_resolved_Wjet   ( cs->_fawjet   [is][ic][ia]     );
		      myChain33_22_pdnnForVBFHVTWZ_VBF_resolved_Diboson( cs->_faDB     [is][ic][ia]     );
		      myChain33_22_pdnnForVBFHVTWZ_VBF_resolved_ttbar  ( cs->_fattbar  [is][ic][ia]     );
		      myChain33_22_pdnnForVBFHVTWZ_VBF_resolved_stop   ( cs->_fastop   [is][ic][ia]     );
		      myChain33_22_pdnnForVBFHVTWZ_VBF_resolved_data   ( cs->_fadata   [is][ic][ia]     );
	      			     					              
		      //myChain33_22_pdnnForVBFHVTWZ_VBF_resolved_VBFHVTWZ( cs->_faVBFHVT [is][ic][ia]     );
		      //cs->_fsignal = cs->_faVBFHVT[is][ic][ia]        ;
		      myChain33_22_pdnnForVBFHVTWZ_VBF_resolved_VBFHVTWZ( cs->_faSignal [is][ic][ia]     );
		    }
		}
	    }
  
	  cs->_fzjet = cs->_fazjet [is][ic][ia];
	  cs->_fwjet = cs->_fawjet [is][ic][ia];
	  cs->_fDB   = cs->_faDB   [is][ic][ia]; 
	  cs->_fttbar= cs->_fattbar[is][ic][ia];
	  cs->_fstop = cs->_fastop [is][ic][ia];
	  
	  cs->_fdata = cs->_fadata [is][ic][ia]; 
	  /*
	    cs->_faRadion    [is][ic][ia]= new TChain("Nominal"); 
	    cs->_faHVT       [is][ic][ia]= new TChain("Nominal");  
	    cs->_faRSG       [is][ic][ia]= new TChain("Nominal"); 
	    
	    cs->_faVBFRadion [is][ic][ia]= new TChain("Nominal"); 
	    cs->_faVBFHVT    [is][ic][ia]= new TChain("Nominal");  
	    cs->_faVBFRSG    [is][ic][ia]= new TChain("Nominal"); 
	  */
	  cs->_fsignal = cs->_faSignal[is][ic][ia];
	  std::cout<<"is, ic, ia = "<<is<<" "<<ic<<" "<<ia<<std::endl;
	  std::cout<<"pointer to cs->_fzjet = <"<<cs->_fzjet<<" >     cs->_fazjet [is][ic][ia]=<"<<cs->_fazjet [is][ic][ia]<<">"<<std::endl;
    }
  else
    {
      cs->_fzjet = cs->_fazjet [is][ic][ia];
      cs->_fwjet = cs->_fawjet [is][ic][ia];
      cs->_fDB   = cs->_faDB   [is][ic][ia]; 
      cs->_fttbar= cs->_fattbar[is][ic][ia];
      cs->_fstop = cs->_fastop [is][ic][ia];
      	     	      
      cs->_fdata = cs->_fadata [is][ic][ia]; 
      /*
	cs->_faRadion    [is][ic][ia]= new TChain("Nominal"); 
	cs->_faHVT       [is][ic][ia]= new TChain("Nominal");  
	cs->_faRSG       [is][ic][ia]= new TChain("Nominal"); 
	
	cs->_faVBFRadion [is][ic][ia]= new TChain("Nominal"); 
	cs->_faVBFHVT    [is][ic][ia]= new TChain("Nominal");  
	cs->_faVBFRSG    [is][ic][ia]= new TChain("Nominal"); 
      */
      cs->_fsignal = cs->_faSignal[is][ic][ia];
      std::cout<<"is, ic, ia = "<<is<<" "<<ic<<" "<<ia<<std::endl;
      std::cout<<"pointer to cs->_fzjet = <"<<cs->_fzjet<<" >     cs->_fazjet [is][ic][ia]=<"<<cs->_fazjet [is][ic][ia]<<">"<<std::endl;

    }

}
