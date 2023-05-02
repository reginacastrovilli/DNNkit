#!/bin/bash
#### 
#### create the tree of directories to be used to store trees with pnn scores for all signals / analysis / channel (12 in total)
#### the starting point (directory) for the tree must exist and must be given as input argument; 
#### the subdirectories already existing are NOT overwritten; the script can be used to complete the tree of subdirectories  
####
startDir=$1
if [ -z "$1" ]
then
    echo "Usage is: ./createOutputDirs.sh startingDirectory"
    exit
fi

if [ ! -d "$startDir" ]
then
    echo "Starting directorY <${startDir}> does not exist; exit"
    exit
fi

mc16aDir=/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16a_VV_2lep_PFlow_UFO_Scores/
mc16dDir=/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16d_VV_2lep_PFlow_UFO_Scores/
mc16eDir=/nfs/kloe/einstein4/HDBS/ReaderOutput/reader_mc16e_VV_2lep_PFlow_UFO_Scores/

#BOOKS=('merged' 'resolved' 'mixed')
#
#for book in "${BOOKS[@]}"; do
#  echo "Book: $book"
#done

suffix=".root"
for analysis in merged resolved
do
    echo "Analysis: $analysis"
    for channel in ggF VBF
    do
	echo "Analysis/channel: $analysis/$channel"
	if [ $channel = ggF ]
	then
	    for signal in RSG Radion HVTWZ
	    do
		echo "Analysis/channel/signal: $analysis/$channel/$signal"
		mySubDir=$analysis/$channel/$signal
		#echo "Going to create dir: $startDir/$mySubDir"
		if [ -d "$startDir/$mySubDir" ] 
		then
		    echo "Directory $startDir/$mySubDir ALREADY exists." 
		else
		    #echo "Directory $startDir/$mySubDir does not exists; let's make it "
		    mkdir -p  $startDir/$mySubDir
		fi
		cd ${mc16aDir}/$mySubDir/
		filelist=`ls *.root`
		cd -
		for inpFile in $filelist ; do
		    inpFileN=${inpFile%"$suffix"}
		    cmd="ln -sf $mc16aDir/$mySubDir/$inpFile $startDir/$mySubDir/${inpFileN}_mc16a.root"
		    #echo $cmd
		    $cmd
		done
		cd ${mc16dDir}/$mySubDir/
		filelist=`ls *.root`
		cd -
		for inpFile in $filelist ; do
		    inpFileN=${inpFile%"$suffix"}
		    cmd="ln -sf $mc16dDir/$mySubDir/$inpFile $startDir/$mySubDir/${inpFileN}_mc16d.root"
		    #echo $cmd
		    $cmd
		done
		cd ${mc16eDir}/$mySubDir/
		filelist=`ls *.root`
		cd -
		for inpFile in $filelist ; do
		    inpFileN=${inpFile%"$suffix"}
		    cmd="ln -sf $mc16eDir/$mySubDir/$inpFile $startDir/$mySubDir/${inpFileN}_mc16e.root"
		    #echo $cmd
		    $cmd
		done
	    done
	else
	    #for signal in VBFRSG VBFRadion VBFHVT
	    for signal in VBFRSG VBFRadion VBFHVTWZ
	    do
		echo "Analysis/channel/signal: $analysis/$channel/$signal"
		mySubDir=$analysis/$channel/$signal
		#echo "Going to create dir: $startDir/$mySubDir"
		if [ -d "$startDir/$mySubDir" ] 
		then
		    echo "Directory $startDir/$mySubDir ALREADY exists." 
		else
		    #echo "Directory $startDir/$mySubDir does not exists; let's make it "
		    mkdir -p  $startDir/$mySubDir
		fi
		cd ${mc16aDir}/$mySubDir/
		filelist=`ls *.root`
		cd -
		for inpFile in $filelist ; do
		    inpFileN=${inpFile%"$suffix"}
		    cmd="ln -sf $mc16aDir/$mySubDir/$inpFile $startDir/$mySubDir/${inpFileN}_mc16a.root"
		    #echo $cmd
		    $cmd
		done
		cd ${mc16dDir}/$mySubDir/
		filelist=`ls *.root`
		cd -
		for inpFile in $filelist ; do
		    inpFileN=${inpFile%"$suffix"}
		    cmd="ln -sf $mc16dDir/$mySubDir/$inpFile $startDir/$mySubDir/${inpFileN}_mc16d.root"
		    #echo $cmd
		    $cmd
		done
		cd ${mc16eDir}/$mySubDir/
		filelist=`ls *.root`
		cd -
		for inpFile in $filelist ; do
		    inpFileN=${inpFile%"$suffix"}
		    cmd="ln -sf $mc16eDir/$mySubDir/$inpFile $startDir/$mySubDir/${inpFileN}_mc16e.root"
		    #echo $cmd
		    $cmd
		done
	    done
	fi
    done
done
echo "Everything done ... "
    
