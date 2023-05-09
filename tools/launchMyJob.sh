#!/bin/bash 

#
defaultMode="buildPDNN"

if [ $# -le 2 ]
then
  echo "$0 requires at least tree arguents: Usage is "
  echo "$0 [analysis] [channel] [signal] [mode=buildPDNN]"
  echo "possible values for [analysis] = merged, resolved"
  echo "possible values for [channel]  = ggF,    VBF"
  echo "possible values for [signal]   = Radion, RSG, HVTWZ"
  echo "possible values for [mode]     = buildPDNN (=default), buildDataset, splitDataset"
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
#########################################################
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
cp runMyJob.sh ${wdir}
cd ${wdir}

echo 'Now working in ' $PWD
echo "Files available are: "
ls -al
#
analysis=$1
channel=$2
signal=$3
log=log_${mode}_${analysis}_${channel}_${signal}_${tag}
#

nohup ./runMyJob.sh ${analysis} ${channel} ${signal} ${mode} > ${log} &
ln -sf $PWD/${log} ${maind}/logs/${log}_${mynodes};
echo "Output logged in " ${maind}/logs/${log}_${mynodes}
sleep 3s
cd ${maind}
echo "Back in " $PWD
echo "All done at `date`"

