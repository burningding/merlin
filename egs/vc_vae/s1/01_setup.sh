#!/bin/bash

if test "$#" -ne 2; then
    echo "################################"
    echo "Usage:"
    echo "$0 <src_speaker> <tgt_speaker>"
    echo ""
    echo "Give a source speaker name eg., bdl"
    echo "Give a target speaker name eg., slt"
    echo "################################"
    exit 1
fi

current_working_dir=$(pwd)
merlin_dir=$(dirname $(dirname $(dirname $current_working_dir)))
exp_dir=${current_working_dir}/exp
data_dir=${current_working_dir}/dataset/arctic

src_speaker=$1
tgt_speaker=$2

voice_name=$12$2

log_dir=${exp_dir}/log
model_dir=${exp_dir}/model
pitch_model_dir=${exp_dir}/pitch_model
rec_feature_dir=${exp_dir}/rec_feature
rec_wav_dir=${exp_dir}/rec_wav
scp_dir=${exp_dir}/scp

mkdir -p ${data_dir}
mkdir -p ${data_dir}/wav
mkdir -p ${data_dir}/wav/$src_speaker
mkdir -p ${data_dir}/wav/$tgt_speaker
mkdir -p ${data_dir}/feature
mkdir -p ${data_dir}/feature/$src_speaker
mkdir -p ${data_dir}/feature/$tgt_speaker

mkdir -p ${exp_dir}
mkdir -p ${log_dir}
mkdir -p ${model_dir}
mkdir -p ${pitch_model_dir}
mkdir -p ${rec_feature_dir}
mkdir -p ${rec_wav_dir}
mkdir -p ${scp_dir}

# create an empty question file
touch ${merlin_dir}/misc/questions/questions-vc.hed

global_config_file=conf/global_settings.cfg

### default settings ###
echo "######################################" > $global_config_file
echo "############# PATHS ##################" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "MerlinDir=${merlin_dir}" >>  $global_config_file
echo "WorkDir=${current_working_dir}" >>  $global_config_file
echo "" >> $global_config_file

echo "######################################" >> $global_config_file
echo "############# PARAMS #################" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "Source=${src_speaker}" >> $global_config_file
echo "Target=${tgt_speaker}" >> $global_config_file
echo "Voice=${voice_name}" >> $global_config_file
echo "" >> $global_config_file

echo "Vocoder=WORLD" >> $global_config_file
echo "SamplingFreq=16000" >> $global_config_file
echo "FileIDList=file_id_list.scp" >> $global_config_file
echo "" >> $global_config_file

echo "######################################" >> $global_config_file
echo "######### No. of files ###############" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "Train=250" >> $global_config_file 
echo "Valid=25" >> $global_config_file 
echo "Test=25" >> $global_config_file 
echo "" >> $global_config_file

echo "Step 1:"
echo "Merlin default voice-conversion settings configured in \"$global_config_file\""
echo "Modify these params as per your data..."
echo "eg., sampling frequency, no. of train files etc.,"
echo "setup done...!"

