void myChain33_22_mc16a_Diboson(TChain *f)
{
  std::string filedir = "/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16a_VV_2lep_PFlow_UFO/fetch/data-MVATree/";
  f->Add((filedir+"/Diboson-0.root").c_str());
  f->Add((filedir+"/Diboson-1.root").c_str());
  f->Add((filedir+"/Diboson-2.root").c_str());
  f->Add((filedir+"/Diboson-3.root").c_str());
  f->Add((filedir+"/Diboson-4.root").c_str());
}
