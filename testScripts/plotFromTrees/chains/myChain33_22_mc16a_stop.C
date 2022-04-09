void myChain33_22_mc16a_stop(TChain *f)
{
  std::string filedir = "/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16a_VV_2lep_PFlow_UFO/fetch/data-MVATree/";
  f->Add((filedir+"/stop-0.root").c_str());
  f->Add((filedir+"/stop-1.root").c_str());
  f->Add((filedir+"/stop-2.root").c_str());
  f->Add((filedir+"/stop-3.root").c_str());
}
