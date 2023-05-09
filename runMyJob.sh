#!/bin/bash 


dryRun=false
defaultMode="buildPDNN"

if [ $# -le 2 ]
then
  echo "$0 requires at least tree arguents: Usage is "
  echo "$0 [analysis] [channel] [signal] [mode=buildPDNN]"
  echo "possible values for [analysis] = merged, resolved"
  echo "possible values for [channel]  = ggF,    VBF"
  echo "possible values for [signal]   = Radion, RSG, HVTWZ"
  echo "possible values for [mode]     = buildPDNN (=default), buildDataset, splitDataSet, ... "
  exit
fi
#


analysis=$1
channel=$2
signal=$3
if [ $# -ge 4 ]
then
  mode=$4
else
  mode=${defaultMode}
fi
#

echo "$0 ===================================== dryRun = " ${dryRun} 
xfound=false
for x in "buildDataset" "splitDataset" "buildPDNN" "buildDNN" "SaveToPkl"; do
    if [ ${x} == ${mode} ]; then
	xfound=true
    fi
done
if [ "$xfound" = false ]; then
    echo "mode = " ${mode} " does not correspond to a known script to run - stop here"
    exit
fi

echo "analysis / channel / signal:       " ${analysis} ${channel} ${signal} 
echo "for macro:                         " ${mode}".py"
echo "running in:                        " $PWD

myCmd="python buildPDNN.py -a ${analysis} -c ${channel} -s ${signal} --doTrain 1"

if [ ${mode} == "buildPDNN" ]; then
  myCmd="python buildPDNN.py -a ${analysis} -c ${channel} -s ${signal} --doTrain 1"
else 
  myCmd="python ${mode}.py -a ${analysis} -c ${channel} -s ${signal}"
fi 

#myCmd="python splitDataset.py -a ${analysis} -c ${channel} -s ${signal}"
echo "About to launch < " $myCmd " >"
if [ "$dryRun" = false ]; then
   echo "We are here ... "
   $myCmd 
fi
#ln -sf ${PWD}/${log} ${maind}/logs/${log}_${mynodes}
    


#

