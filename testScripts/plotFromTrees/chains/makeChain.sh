#!/bin/bash 

rm -f myChain.C 
touch myChain.C

for signal in RSG Radion HVTWZ
do
    for channel in ggF VBF
    do
	if [ $channel == "VBF" ]; then
	    signal="VBF${signal}"
	    echo "Redefining signal as ${signal} since channel is ${channel}"
	else
	    echo "Signal is ${signal} since channel is ${channel}"
	fi
	for analysis in merged resolved
	do
	    for sample in Zjet Wjet Diboson ttbar stop data ${signal}
	    do
		echo "void myChain33_22_pdnnFor${signal}_${channel}_${analysis}_${sample}(TChain *f)" >>myChain.C
		echo "{" >>myChain.C
		echo "  std::string filedir;" >>myChain.C
		for mcType in mc16a mc16d mc16e
		do
		    #fileLocation="/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16d_VV_2lep_PFlow_UFO/fetch/data-MVATree/"
		    fileLocation="/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_${mcType}_VV_2lep_PFlow_UFO_Scores/${analysis}/${channel}/${signal}/"
		    echo "making chain for files in ${fileLocation}"
		    echo "  filedir = \""$fileLocation"\";" >>myChain.C 
		    #ls -la $fileLocation
		    filelist=`ls ${fileLocation}/ | grep root | grep "Scores_"${sample} `
		    for inpFile in $filelist ; do
			#echo ... adding ${inpFile}
			echo "  f->Add((filedir+\"${inpFile}\").c_str());" >>myChain.C
			#echo "  f->Add((\"${inpFile}\").c_str());" >>myChain.C
		    done
		    #mv myChain.C myChain33_22_pdnnFor${signal}_${channel}_${analysis}_${mcType}_${sample}.C
		done
		echo "}">>myChain.C
	    done
	done
    done
done
mv myChain.C myChain33_22_pdnnScores.C 

echo "done"


