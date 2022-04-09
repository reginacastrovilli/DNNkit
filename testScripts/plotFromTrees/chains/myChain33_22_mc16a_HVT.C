void myChain33_22_mc16a_HVT(TChain *f)
{
  std::string filedir = "/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16a_VV_2lep_PFlow_UFO/fetch/data-MVATree/";
  f->Add((filedir+"/HVTWZ-0.root").c_str());
  f->Add((filedir+"/HVTWZ-1.root").c_str());
  f->Add((filedir+"/HVTWZ-2.root").c_str());
  f->Add((filedir+"/HVTWZ-3.root").c_str());
  f->Add((filedir+"/HVTWZ-4.root").c_str());
}
