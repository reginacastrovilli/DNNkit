void myChain33_22_mc16a_Radion(TChain *f)
{
  std::string filedir = "/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16a_VV_2lep_PFlow_UFO/fetch/data-MVATree/";
  f->Add((filedir+"/Radion-0.root").c_str());
  f->Add((filedir+"/Radion-1.root").c_str());
  f->Add((filedir+"/Radion-2.root").c_str());
  f->Add((filedir+"/Radion-3.root").c_str());
}
