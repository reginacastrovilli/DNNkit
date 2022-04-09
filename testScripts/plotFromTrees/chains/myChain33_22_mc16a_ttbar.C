void myChain33_22_mc16a_ttbar(TChain *f)
{
  std::string filedir = "/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16a_VV_2lep_PFlow_UFO/fetch/data-MVATree/";
  f->Add((filedir+"/ttbar-0.root").c_str());
  f->Add((filedir+"/ttbar-10.root").c_str());
  f->Add((filedir+"/ttbar-1.root").c_str());
  f->Add((filedir+"/ttbar-2.root").c_str());
  f->Add((filedir+"/ttbar-3.root").c_str());
  f->Add((filedir+"/ttbar-4.root").c_str());
  f->Add((filedir+"/ttbar-5.root").c_str());
  f->Add((filedir+"/ttbar-6.root").c_str());
  f->Add((filedir+"/ttbar-7.root").c_str());
  f->Add((filedir+"/ttbar-8.root").c_str());
  f->Add((filedir+"/ttbar-9.root").c_str());
}
