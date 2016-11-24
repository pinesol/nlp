#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: ./generate_pbs.sh experiment_name [flags]"
    echo "e.g.: ./generate_pbs.sh experiment_1 --data_dir=data --train_dir=save/perptest --use_attention=false --num_layers=1 --size=128"
    exit
fi

#Generate_pbs.sh
experiment_name=$1 #eg experiment_1


echo "#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=20:00:00
#PBS -l mem=25GB
#PBS -N ${experiment_name}
#PBS -j oe
#PBS -M alex.pine@nyu.edu
#PBS -m ae

cd /home/akp258/nlp/hw4

module purge
module load pillow/intel/2.7.0
#module load tensorflow/python2.7/20160721
module load tensorflow/python2.7/20161029
module load scipy/intel/0.18.0
module load nltk/3.0.2

python translate.py ${@:2}

" > ${experiment_name}.pbs
