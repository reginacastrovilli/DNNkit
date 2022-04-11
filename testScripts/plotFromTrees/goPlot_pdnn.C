{
  /// create pointers to empty chains 
	gROOT->ProcessLine(".x initToNullChains_33_22_pdnn.C(cs)");

	/////////////// SIGNAL REGIONS ///////////////////////////////////////////////////////////

	///////// ZZ  VBF SR ----------------------------------
	
	// Merged HP
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, cs, \"Pass_MergHP_VBF_ZZ_SR==1\"       , \"Run2_AllowOverlap_ZZ_VBF_MHP_SR\"     , true, false, \"RSG\")");
	
	// Merged LP 
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_VBF_ZZ_SR==1\"       , \"Run2_AllowOverlap_ZZ_VBF_MLP_SR\"     , true, false, \"RSG\")");
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_resolved_ZZ_m\",100,0.,3500.,false, true, \"Pass_Res_VBF_ZZ_SR==1\"          , \"Run2_AllowOverlap_ZZ_VBF_Res_SR\"     , true, false, \"RSG\")");

	// ZZ  ggF/DY SR ----------------------------------
	
	// Merged HP
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_GGF_ZZ_Tag_SR==1\"   , \"Run2_AllowOverlap_ZZ_GGF_MHPTag_SR\"   , true, false, \"RSG\")");
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_GGF_ZZ_Untag_SR==1\" , \"Run2_AllowOverlap_ZZ_GGF_MHPnoT_SR\"   , true, false)");	
	// Merged LP 
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_GGF_ZZ_Tag_SR==1\"   , \"Run2_AllowOverlap_ZZ_GGF_MLPTag_SR\"   , true, false, \"RSG\")");
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_GGF_ZZ_Untag_SR==1\" , \"Run2_AllowOverlap_ZZ_GGF_MLPnoT_SR\"   , true, false, \"RSG\")"); 
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_resolved_ZZ_m\",100,0.,3500.,false, true, \"Pass_Res_GGF_ZZ_Tag_SR==1\"      , \"Run2_AllowOverlap_ZZ_GGF_ResTag_SR\"   , true, false, \"RSG\")");
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_resolved_ZZ_m\",100,0.,3500.,false, true, \"Pass_Res_GGF_ZZ_Untag_SR==1\"    , \"Run2_AllowOverlap_ZZ_GGF_ResnoT_SR\"   , true, false, \"RSG\")");

	///////// WZ  VBF SR ----------------------------------
	
	// Merged HP
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_VBF_WZ_SR==1\"       , \"Run2_AllowOverlap_WZ_VBF_MHP_SR\"     , true, false, \"RSG\")");
	// Merged LP
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_VBF_WZ_SR==1\"       , \"Run2_AllowOverlap_WZ_VBF_MLP_SR\"     , true, false, \"RSG\")");
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_resolved_WZ_m\",100,0.,3500.,false, true, \"Pass_Res_VBF_WZ_SR==1\"          , \"Run2_AllowOverlap_WZ_VBF_Res_SR\"     , true, false, \"RSG\")");

	///////// WZ  ggF/DY SR ----------------------------------
	
	// Merged HP
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_GGF_WZ_SR==1\"       , \"Run2_AllowOverlap_WZ_GGF_MHP_SR\"      , true, false, \"HVT\")");
	// Merged LP 
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_GGF_WZ_SR==1\"       , \"Run2_AllowOverlap_WZ_GGF_MLP_SR\"      , true, false, \"HVT\")");
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_resolved_WZ_m\",100,0.,3500.,false, true, \"Pass_Res_GGF_WZ_SR==1\"          , \"Run2_AllowOverlap_WZ_GGF_Res_SR\"      , true, false, \"HVT\")");

	/////////////// Z+jets CONTROL REGIONS ///////////////////////////////////////////////////////////

	///////// ZZ  VBF ZCR ----------------------------------
	
	// Merged HP
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_VBF_ZZ_ZCR==1\"       , \"Run2_AllowOverlap_ZZ_VBF_MHP_ZCR\"   , true, false, \"RSG\")");
	// Merged LP 
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_VBF_ZZ_ZCR==1\"       , \"Run2_AllowOverlap_ZZ_VBF_MLP_ZCR\"   , true, false, \"RSG\")");
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_resolved_ZZ_m\",100,0.,3500.,false, true, \"Pass_Res_VBF_ZZ_ZCR==1\"          , \"Run2_AllowOverlap_ZZ_VBF_Res_ZCR\"   , true, false, \"RSG\")");

	// ZZ  ggF/DY ZCR ----------------------------------
	
	// Merged HP
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_GGF_ZZ_Tag_ZCR==1\"   , \"Run2_AllowOverlap_ZZ_GGF_MHPTag_ZCR\" , true, false, \"RSG\")");
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_GGF_ZZ_Untag_ZCR==1\" , \"Run2_AllowOverlap_ZZ_GGF_MHPnoT_ZCR\" , true, false, \"RSG\")");	
	// Merged LP 
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_GGF_ZZ_Tag_ZCR==1\"   , \"Run2_AllowOverlap_ZZ_GGF_MLPTag_ZCR\" , true, false, \"RSG\")");
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_GGF_ZZ_Untag_ZCR==1\" , \"Run2_AllowOverlap_ZZ_GGF_MLPnoT_ZCR\" , true, false, \"RSG\")"); 
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_resolved_ZZ_m\",100,0.,3500.,false, true, \"Pass_Res_GGF_ZZ_Tag_ZCR==1\"      , \"Run2_AllowOverlap_ZZ_GGF_ResTag_ZCR\" , true, false, \"RSG\")");
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_resolved_ZZ_m\",100,0.,3500.,false, true, \"Pass_Res_GGF_ZZ_Untag_ZCR==1\"    , \"Run2_AllowOverlap_ZZ_GGF_ResnoT_ZCR\" , true, false, \"RSG\")");

	///////// WZ  VBF ZCR ----------------------------------
	
	// Merged HP
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_VBF_WZ_ZCR==1\"       , \"Run2_AllowOverlap_WZ_VBF_MHP_ZCR\"   , true, false, \"RSG\")");
	// Merged LP
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_VBF_WZ_ZCR==1\"       , \"Run2_AllowOverlap_WZ_VBF_MLP_ZCR\"   , true, false, \"RSG\")");
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_resolved_WZ_m\",100,0.,3500.,false, true, \"Pass_Res_VBF_WZ_ZCR==1\"          , \"Run2_AllowOverlap_WZ_VBF_Res_ZCR\"   , true, false, \"RSG\")");

	///////// WZ  ggF/DY ZCR ----------------------------------
	
	// Merged HP
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_GGF_WZ_ZCR==1\"       , \"Run2_AllowOverlap_WZ_GGF_MHP_ZCR\"    , true, false, \"HVT\")");
	// Merged LP 
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_GGF_WZ_ZCR==1\"       , \"Run2_AllowOverlap_WZ_GGF_MLP_ZCR\"    , true, false, \"HVT\")");
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022_pdnn.C(\"X_resolved_WZ_m\",100,0.,3500.,false, true, \"Pass_Res_GGF_WZ_ZCR==1\"          , \"Run2_AllowOverlap_WZ_GGF_Res_ZCR\"    , true, false, \"HVT\")");
	
}
