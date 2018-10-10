#!/bin/bash

setup_data=true
train_vc=true
run_vc=true

if test "$#" -ne 2; then
    src_speaker="bdl"
    tgt_speaker="slt"
else
    src_speaker=$1
    tgt_speaker=$2
    setup_data=false
fi

# setup directory structure and download data
if [ "$setup_data" = true ]; then
    # download demo data (300 utterances)
    wget http://104.131.174.95/downloads/voice_conversion/bdl_arctic.zip
    wget http://104.131.174.95/downloads/voice_conversion/slt_arctic.zip
    wget http://104.131.174.95/downloads/voice_conversion/rms_arctic.zip
    wget http://104.131.174.95/downloads/voice_conversion/clb_arctic.zip

    # unzip files
    unzip -q bdl_arctic.zip
    unzip -q slt_arctic.zip
    unzip -q rms_arctic.zip
    unzip -q clb_arctic.zip

    mkdir -p dataset
    mkdir -p dataset/bdl
    mkdir -p dataset/slt
    mkdir -p dataset/rms
    mkdir -p dataset/clb

    # copy data
    mv bdl_arctic/wav dataset/bdl/wav
    mv slt_arctic/wav dataset/slt/wav
    mv rms_arctic/wav dataset/rms/wav
    mv clb_arctic/wav dataset/clb/wav

    rm -rf bdl_arctic slt_arctic rms_arctic clb_arctic
    rm -rf bdl_arctic.zip slt_arctic.zip rms_arctic.zip clb_arctic.zip
fi


