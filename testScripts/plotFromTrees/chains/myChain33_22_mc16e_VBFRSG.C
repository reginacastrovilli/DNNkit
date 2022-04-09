void myChain33_22_mc16e_VBFRSG(TChain *f)
{
  std::string filedir = "/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16e_VV_2lep_PFlow_UFO/fetch/data-MVATree/";
  f->Add((filedir+"/VBFRSG-0.root").c_str());
  f->Add((filedir+"/VBFRSG-1.root").c_str());
  f->Add((filedir+"/VBFRSG-2.root").c_str());
  f->Add((filedir+"/VBFRSG-3.root").c_str());
  f->Add((filedir+"/VBFRSG-4.root").c_str());
}
