#ifndef CHAINS_33_22_PDNN_H
#define CHAINS_33_22_PDNN_H

#include "TChain.h"
class chainSet_33_22_pdnn {
 public:

  chainSet_33_22_pdnn()
    {
      for (unsigned int is=0; is<3; ++is)
	for (unsigned int ic=0; ic<2; ++ic)
	  for (unsigned int ia=0; ia<2; ++ia)
	    {
	      _fazjet [is][ic][ia] =NULL;
	      _fawjet [is][ic][ia] =NULL;
	      _faDB   [is][ic][ia] =NULL;       
	      _fattbar[is][ic][ia] =NULL;       
	      _fastop [is][ic][ia] =NULL;       
	      _fadata [is][ic][ia] =NULL;       
	      /*
		_faRadion   [is][ic][ia]=NULL; 
		_faHVT      [is][ic][ia]=NULL; 
		_faRSG      [is][ic][ia]=NULL; 
		_faVBFRadion[is][ic][ia]=NULL; 
		_faVBFHVT   [is][ic][ia]=NULL; 
		_faVBFRSG   [is][ic][ia]=NULL; 
	      */
	      _faSignal [is][ic][ia] =NULL; 
	    }			
      
      _fzjet          =NULL;
      _fDB            =NULL; 
      _fwjet          =NULL;
      _fttbar         =NULL; 
      _fstop          =NULL; 
      _fdata          =NULL; 
      _fsignal        =NULL; 
      
      /*
	_fRadion        =NULL; 
	_fHVT           =NULL; 
	_fRSG           =NULL; 
	_fVBFRadion     =NULL; 
	_fVBFHVT        =NULL; 
	_fVBFRSG        =NULL; 
      */
    }
  TChain* _fzjet;
  TChain* _fDB  ; 
  TChain* _fwjet  ;
  TChain* _fttbar ; 
  TChain* _fstop  ; 
  TChain* _fdata; 
  TChain* _fsignal; 
  
  /*
    TChain* _fRadion; 
    TChain* _fHVT; 
    TChain* _fRSG; 
    TChain* _fVBFRadion; 
    TChain* _fVBFHVT; 
    TChain* _fVBFRSG; 
  */
  
  TChain* _fadata[3][2][2]; 
  TChain* _fazjet[3][2][2];
  TChain* _faDB[3][2][2]  ; 
  TChain* _fawjet[3][2][2]  ;
  TChain* _fattbar[3][2][2] ; 
  TChain* _fastop[3][2][2]  ; 
  
  /*
    TChain* _fRadion[3][2][2]; 
    TChain* _fHVT[3][2][2]; 
    TChain* _fRSG[3][2][2]; 
    TChain* _fVBFRadion[3][2][2]; 
    TChain* _fVBFHVT[3][2][2]; 
    TChain* _fVBFRSG[3][2][2]; 
  */
  TChain* _faSignal[3][2][2];


  
};

#endif 
