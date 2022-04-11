#!/bin/bash 

defaultExc="NothingToExclude"
sample=$1
excSample=$2
if [ -z "$sample" ]
then
    echo "Usage is: ./makeChain.sh data"
    echo "Usage is: ./makeChain.sh Zjet"
fi
if [ -z "$excSample" ]
then
   excSample=$defaultExc
fi
rm -f myChain.C 
touch myChain.C


#fileLocation="/eos/user/c/chiodini/CONDOR_output/ZV2Lep_TCC_EMTopo_Nopass/fetch/data-MVATree"
#fileLocation="/eos/user/m/mcentonz/CxAODReader_output/out_ZV2Lep_TCC_EMTopo/fetch/data-MVATree/"
#####fileLocation="/nfs/kloe/einstein4/HDBS/PDNNTestAGS/ntuples/"
#fileLocation="/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16a_VV_2lep_PFlow_UFO/fetch/data-MVATree/"
#fileLocation="/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16d_VV_2lep_PFlow_UFO/fetch/data-MVATree/"
fileLocation="/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16*_VV_2lep_PFlow_UFO/fetch/data-MVATree/"


#echo "void myChain33_22_mc16a_${sample}(TChain *f)" >>myChain.C
#echo "void myChain33_22_mc16d_${sample}(TChain *f)" >>myChain.C
#echo "void myChain33_22_mc16e_${sample}(TChain *f)" >>myChain.C
echo "void myChain33_22_mc16_${sample}(TChain *f)" >>myChain.C

echo "{" >>myChain.C
echo "  std::string filedir = \""$fileLocation"\";" >>myChain.C 
ls -la $fileLocation
filelist=`ls ${fileLocation}/*root | grep ${sample} | grep -v ${excSample}`
for inpFile in $filelist ; do
    echo ... adding ${inpFile}
    #echo "  f->Add((filedir+\"/${inpFile}\").c_str());" >>myChain.C
    echo "  f->Add((\"/${inpFile}\").c_str());" >>myChain.C
done
echo "}" >>myChain.C
#mv myChain.C myChain33_22_mc16a"_"${sample}.C
#mv myChain.C myChain33_22_mc16d"_"${sample}.C
#mv myChain.C myChain33_22_mc16e"_"${sample}.C
mv myChain.C myChain33_22_mc16"_"${sample}.C

echo "done"


