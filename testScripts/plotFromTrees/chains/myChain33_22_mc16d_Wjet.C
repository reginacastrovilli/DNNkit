void myChain33_22_mc16d_Wjet(TChain *f)
{
  std::string filedir = "/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16d_VV_2lep_PFlow_UFO/fetch/data-MVATree/";
  f->Add((filedir+"/Wjets-0.root").c_str());
  f->Add((filedir+"/Wjets-10.root").c_str());
  f->Add((filedir+"/Wjets-11.root").c_str());
  f->Add((filedir+"/Wjets-12.root").c_str());
  f->Add((filedir+"/Wjets-13.root").c_str());
  f->Add((filedir+"/Wjets-14.root").c_str());
  f->Add((filedir+"/Wjets-15.root").c_str());
  f->Add((filedir+"/Wjets-16.root").c_str());
  f->Add((filedir+"/Wjets-1.root").c_str());
  f->Add((filedir+"/Wjets-2.root").c_str());
  f->Add((filedir+"/Wjets-3.root").c_str());
  f->Add((filedir+"/Wjets-4.root").c_str());
  f->Add((filedir+"/Wjets-5.root").c_str());
  f->Add((filedir+"/Wjets-6.root").c_str());
  f->Add((filedir+"/Wjets-7.root").c_str());
  f->Add((filedir+"/Wjets-8.root").c_str());
  f->Add((filedir+"/Wjets-9.root").c_str());
}
