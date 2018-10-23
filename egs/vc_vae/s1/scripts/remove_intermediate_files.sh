#!/bin/bash

if test "$#" -ne 1; then
    echo "Usage: ./scripts/remove_intermediate_files.sh conf/global_settings.cfg"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $1
fi

###################################################
######## remove intermediate synth files ##########
###################################################

current_working_dir=$(pwd)

exp_dir=exp/arctic
rec_wav_dir=${exp_dir}/rec_wav/${Voice}

shopt -s extglob

if [ -d "$rec_wav_dir" ]; then
    cd ${rec_wav_dir}
    rm -f *.!(wav)
fi

cd ${current_working_dir}
