#!/bin/bash 

dryRun=false

analysis=$1
channel=$2
signal=$3
#

echo "ruMyJob1 ......... dryRun = " ${dryRun} " analysis / channel / signal " ${analysis} ${channel} ${signal}
echo "In " $PWD

myCmd="python buildPDNN.py -a ${analysis} -c ${channel} -s ${signal} --doTrain 1"
echo "About to launch < " $myCmd " >"
if [ "$dryRun" = false ]; then
   echo "We are here ... "
   $myCmd 
fi
#ln -sf ${PWD}/${log} ${maind}/logs/${log}_${mynodes}
    


#

