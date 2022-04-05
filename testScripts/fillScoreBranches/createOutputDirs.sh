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


#BOOKS=('merged' 'resolved' 'mixed')
#
#for book in "${BOOKS[@]}"; do
#  echo "Book: $book"
#done

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
	    done
	fi
    done
done
echo "Everything done ... "
    
