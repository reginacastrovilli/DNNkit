#include "chains/myChain33_22_mc16a_Zjet.C"
#include "chains/myChain33_22_mc16a_Wjet.C"
#include "chains/myChain33_22_mc16a_Diboson.C"
#include "chains/myChain33_22_mc16a_ttbar.C"
#include "chains/myChain33_22_mc16a_stop.C"
#include "chains/myChain33_22_mc16a_data.C"
#include "chains/myChain33_22_mc16a_VBFRSG.C"
#include "chains/myChain33_22_mc16a_VBFRadion.C"
#include "chains/myChain33_22_mc16a_VBFHVT.C"
#include "chains/myChain33_22_mc16a_RSG.C"
#include "chains/myChain33_22_mc16a_Radion.C"
#include "chains/myChain33_22_mc16a_HVT.C"

#include "chains/myChain33_22_mc16d_Zjet.C"
#include "chains/myChain33_22_mc16d_Wjet.C"
#include "chains/myChain33_22_mc16d_Diboson.C"
#include "chains/myChain33_22_mc16d_ttbar.C"
#include "chains/myChain33_22_mc16d_stop.C"
#include "chains/myChain33_22_mc16d_data.C"
#include "chains/myChain33_22_mc16d_VBFRSG.C"
#include "chains/myChain33_22_mc16d_VBFRadion.C"
#include "chains/myChain33_22_mc16d_VBFHVT.C"
#include "chains/myChain33_22_mc16d_RSG.C"
#include "chains/myChain33_22_mc16d_Radion.C"
#include "chains/myChain33_22_mc16d_HVT.C"

#include "chains/myChain33_22_mc16e_Zjet.C"
#include "chains/myChain33_22_mc16e_Wjet.C"
#include "chains/myChain33_22_mc16e_Diboson.C"
#include "chains/myChain33_22_mc16e_ttbar.C"
#include "chains/myChain33_22_mc16e_stop.C"
#include "chains/myChain33_22_mc16e_data.C"
#include "chains/myChain33_22_mc16e_VBFRSG.C"
#include "chains/myChain33_22_mc16e_VBFRadion.C"
#include "chains/myChain33_22_mc16e_VBFHVT.C"
#include "chains/myChain33_22_mc16e_RSG.C"
#include "chains/myChain33_22_mc16e_Radion.C"
#include "chains/myChain33_22_mc16e_HVT.C"

#include <string>

#include "TChain.h"
TChain* _fzjet;
TChain* _fDB  ; 
TChain* _fwjet  ;
TChain* _fttbar ; 
TChain* _fstop  ; 
TChain* _fRadion; 
TChain* _fHVT; 
TChain* _fRSG; 
TChain* _fVBFRadion; 
TChain* _fVBFHVT; 
TChain* _fVBFRSG; 
TChain* _fdata; 

void initChains_33_22(std::string dataset="")
{

  bool mc16a = false;
  bool mc16d = false;
  bool mc16e = false;


  if (dataset=="run2" || dataset=="all")
    {
      mc16a = true;
      mc16d = true;
      mc16e = true;
    }
  else if (dataset=="mc16a")
    {
      mc16a = true;
    }
  else if (dataset=="mc16d")
    {
      mc16d = true;
    }
  else if (dataset=="mc16e")
    {
      mc16e = true;
    }
  else
    {
      std::cout<<"Run with:  .x initChains_33_22.C(dataset)        "<<std::endl;
      std::cout<<"Allowed values for dataset = mc16a, mc16d, mc16e, run2"<<std::endl;
      return;
    }

  _fzjet  = new TChain("Nominal");;
  _fwjet  = new TChain("Nominal");;
  _fDB    = new TChain("Nominal");; 
  _fttbar = new TChain("Nominal");;
  _fstop  = new TChain("Nominal");;

  _fdata   = new TChain("Nominal"); 
  
  _fRadion = new TChain("Nominal"); 
  _fHVT    = new TChain("Nominal");  
  _fRSG    = new TChain("Nominal"); 

  _fVBFRadion = new TChain("Nominal"); 
  _fVBFHVT    = new TChain("Nominal");  
  _fVBFRSG    = new TChain("Nominal"); 


  if (mc16a) {
    myChain33_22_mc16a_Zjet   ( _fzjet        );
    myChain33_22_mc16a_Wjet   ( _fwjet        );
    myChain33_22_mc16a_Diboson( _fDB          );
    myChain33_22_mc16a_ttbar  ( _fttbar       );
    myChain33_22_mc16a_stop   ( _fstop        );
    myChain33_22_mc16a_data   ( _fdata        );
    
    myChain33_22_mc16a_Radion   ( _fRadion        );
    myChain33_22_mc16a_RSG      ( _fRSG           );
    myChain33_22_mc16a_HVT      ( _fHVT           );
    myChain33_22_mc16a_VBFRadion   ( _fVBFRadion        );
    myChain33_22_mc16a_VBFRSG      ( _fVBFRSG           );
    myChain33_22_mc16a_VBFHVT      ( _fVBFHVT           );
  }
  if (mc16d) {
    myChain33_22_mc16d_Zjet   ( _fzjet        );
    myChain33_22_mc16d_Wjet   ( _fwjet        );
    myChain33_22_mc16d_Diboson( _fDB          );
    myChain33_22_mc16d_ttbar  ( _fttbar       );
    myChain33_22_mc16d_stop   ( _fstop        );
    myChain33_22_mc16d_data   ( _fdata        );
    
    myChain33_22_mc16d_Radion   ( _fRadion        );
    myChain33_22_mc16d_RSG      ( _fRSG           );
    myChain33_22_mc16d_HVT      ( _fHVT           );
    myChain33_22_mc16d_VBFRadion   ( _fVBFRadion        );
    myChain33_22_mc16d_VBFRSG      ( _fVBFRSG           );
    myChain33_22_mc16d_VBFHVT      ( _fVBFHVT           );      
    }

  if (mc16e) {
    myChain33_22_mc16e_Zjet   ( _fzjet        );
    myChain33_22_mc16e_Wjet   ( _fwjet        );
    myChain33_22_mc16e_Diboson( _fDB          );
    myChain33_22_mc16e_ttbar  ( _fttbar       );
    myChain33_22_mc16e_stop   ( _fstop        );
    myChain33_22_mc16e_data   ( _fdata        );
    
    myChain33_22_mc16e_Radion   ( _fRadion        );
    myChain33_22_mc16e_RSG      ( _fRSG           );
    myChain33_22_mc16e_HVT      ( _fHVT           );
    myChain33_22_mc16e_VBFRadion   ( _fVBFRadion        );
    myChain33_22_mc16e_VBFRSG      ( _fVBFRSG           );
    myChain33_22_mc16e_VBFHVT      ( _fVBFHVT           );      
    }


}
