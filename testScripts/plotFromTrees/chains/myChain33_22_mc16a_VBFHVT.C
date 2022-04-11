void myChain33_22_mc16a_VBFHVT(TChain *f)
{
  std::string filedir = "/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16a_VV_2lep_PFlow_UFO/fetch/data-MVATree/";
  f->Add((filedir+"/VBFHVTWZ-0.root").c_str());
  f->Add((filedir+"/VBFHVTWZ-1.root").c_str());
  f->Add((filedir+"/VBFHVTWZ-2.root").c_str());
}
