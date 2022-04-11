{
	gROOT->ProcessLine(".x initChains_33_22.C\(\"run2\")");

	/////////////// SIGNAL REGIONS ///////////////////////////////////////////////////////////

	///////// ZZ and WZ VBF SR ----------------------------------
	
	// Merged HP
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_VBF_ZZ_SR==1||Pass_MergHP_VBF_WZ_SR==1\"       , \"Run2_AllowOverlap_ZZandWZ_VBF_MHP_SR\"     , true, false)");
	// Merged LP 
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_VBF_ZZ_SR==1||Pass_MergLP_VBF_WZ_SR==1\"       , \"Run2_AllowOverlap_ZZandWZ_VBF_MLP_SR\"     , true, false)");
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_resolved_ZZ_m\",100,0.,3500,false, true, \"Pass_Res_VBF_ZZ_SR==1||Pass_Res_VBF_WZ_SR==1\"          , \"Run2_AllowOverlap_ZZandWZ_VBF_Res_SR\"     , true, false)");

	// ZZ and WZ ggF/DY SR ----------------------------------
	
	// Merged HP tag or untag
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_GGF_ZZ_Tag_SR==1||Pass_MergHP_GGF_ZZ_Untag_SR==1||Pass_MergHP_GGF_WZ_SR==1\"   , \"Run2_AllowOverlap_ZZandWZ_GGF_MHP_SR\"   , true, false)");
	// Merged LP tag or untag
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_GGF_ZZ_Tag_SR==1||Pass_MergLP_GGF_ZZ_Untag_SR==1||Pass_MergLP_GGF_WZ_SR==1\"   , \"Run2_AllowOverlap_ZZandWZ_GGF_MLP_SR\"   , true, false)");
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_resolved_ZZ_m\",100,0.,3500.,false, true, \"Pass_Res_GGF_ZZ_Tag_SR==1||Pass_Res_GGF_ZZ_Untag_SR==1||Pass_Res_GGF_WZ_SR==1\"      , \"Run2_AllowOverlap_ZZandWZ_GGF_Res_SR\"   , true, false)");


	/////////////// Z+jets CONTROL REGIONS ///////////////////////////////////////////////////////////

	///////// ZZ ans WZ  VBF ZCR ----------------------------------
	
	// Merged HP
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_VBF_ZZ_ZCR==1||Pass_MergHP_VBF_WZ_ZCR==1\"       , \"Run2_AllowOverlap_ZZandWZ_VBF_MHP_ZCR\"   , true, false)");
	// Merged LP 
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_VBF_ZZ_ZCR==1||Pass_MergLP_VBF_WZ_ZCR==1\"       , \"Run2_AllowOverlap_ZZandWZ_VBF_MLP_ZCR\"   , true, false)");
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_resolved_ZZ_m\",100,0.,3500.,false, true, \"Pass_Res_VBF_ZZ_ZCR==1||Pass_Res_VBF_WZ_ZCR==1\"          , \"Run2_AllowOverlap_ZZandWZ_VBF_Res_ZCR\"   , true, false)");

	// ZZ  and WZ ggF/DY ZCR ----------------------------------
	
	// Merged HP
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergHP_GGF_ZZ_Tag_ZCR==1||Pass_MergHP_GGF_ZZ_Untag_ZCR==1||Pass_MergHP_GGF_WZ_ZCR==1\"   , \"Run2_AllowOverlap_ZZandWZ_GGF_MHP_ZCR\" , true, false)");
	// Merged LP 
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_boosted_m\"    ,100,0.,3500.,false, true, \"Pass_MergLP_GGF_ZZ_Tag_ZCR==1||Pass_MergLP_GGF_ZZ_Untag_ZCR==1||Pass_MergLP_GGF_WZ_ZCR==1\"   , \"Run2_AllowOverlap_ZZandWZ_GGF_MLP_ZCR\" , true, false)");
	// Resolved  
	gROOT->ProcessLine(".x makeMyQuickPlot2022.C(\"X_resolved_ZZ_m\",100,0.,3500.,false, true, \"Pass_Res_GGF_ZZ_Tag_ZCR==1||Pass_Res_GGF_ZZ_Untag_ZCR==1||Pass_Res_GGF_WZ_ZCR==1\"      , \"Run2_AllowOverlap_ZZandWZ_GGF_Res_ZCR\" , true, false)");

}
