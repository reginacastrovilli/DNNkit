void myChain33_22_mc16e_VBFRadion(TChain *f)
{
  std::string filedir = "/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16e_VV_2lep_PFlow_UFO/fetch/data-MVATree/";
  f->Add((filedir+"/VBFRadion-0.root").c_str());
  f->Add((filedir+"/VBFRadion-1.root").c_str());
  f->Add((filedir+"/VBFRadion-2.root").c_str());
  f->Add((filedir+"/VBFRadion-3.root").c_str());
  f->Add((filedir+"/VBFRadion-4.root").c_str());
}
