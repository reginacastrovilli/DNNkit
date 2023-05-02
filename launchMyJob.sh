#!/bin/bash 

dryRun=true

#
maind=$PWD
#
tag=`date +"%Y%m%d_%H_%M_%S"`
mynode=`cat /proc/sys/kernel/hostname`
mynodes=${mynode:0:5}

echo "Starting at " ${date} " on node " ${mynodes} " in main directory " ${maind} "\n"

wdir=/scratch/spagnolo/${tag}
mkdir -p ${wdir}
#
cp *.py ${wdir}
cp *.ini ${wdir}
cp runMyJob1.sh ${wdir}
cd ${wdir}

echo 'Now working in ' $PWD
echo "Files available are: "
ls -al
#
analysis=$1
channel=$2
signal=$3
log=log_${analysis}_${channel}_${signal}_${tag}
#

nohup ./runMyJob1.sh ${analysis} ${channel} ${signal} > ${log} &
ln -sf $PWD/${log} ${maind}/logs/${log}_${mynodes};
echo "Output logged in " ${maind}/logs/${log}_${mynodes}
sleep 3s
cd ${maind}
echo "Back in " $PWD
echo "All done at `date`"

