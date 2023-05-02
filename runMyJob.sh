#!/bin/bash 

dryRun=false
maind=$PWD 

analysis=$1
channel=$2
signal=$3
#
tag=`date +"%Y%m%d_%H_%M_%S"`
echo 'tag = ' ${tag}
mynode=`cat /proc/sys/kernel/hostname`
mynodes=${mynode:0:5}
wdir=/scratch/spagnolo/${tag}
echo 'wdir = ' ${wdir}
#
mkdir ${wdir}
log=log_${analysis}_${channel}_${signal}_${tag}
cp *.py ${wdir}/
cp *.ini ${wdir}/
cd ${wdir}
echo 'Now working in ' $PWD
echo 'Files available ...'
ls -al 
echo 'Now working in ' $PWD >${log}
echo 'Files available ...' >>${log}
ls -al >>${log}
myCmd="python buildPDNN.py -a ${analysis} -c ${channel} -s ${signal} --doTrain 1 > ${log}"
echo $myCmd
if [ "$dryRun" = false ]; then
   echo "We are here ... "
   $myCmd 
fi
ln -sf ${PWD}/${log} ${maind}/logs/${log}_${mynodes}
    


#

