void myChain33_22_mc16e_RSG(TChain *f)
{
  std::string filedir = "/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16e_VV_2lep_PFlow_UFO/fetch/data-MVATree/";
  f->Add((filedir+"/RSG-0.root").c_str());
  f->Add((filedir+"/RSG-1.root").c_str());
  f->Add((filedir+"/RSG-2.root").c_str());
  f->Add((filedir+"/RSG-3.root").c_str());
}
