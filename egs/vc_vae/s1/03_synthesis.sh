#!/bin/bash -e

global_config_file=conf/global_settings.cfg
source $global_config_file

exp_dir=exp/arctic
scp_dir=${exp_dir}/scp/test.scp
rec_feat_dir=${exp_dir}/rec_feature/${Voice}
rec_wav_dir=${exp_dir}/rec_wav/${Voice}

### Synthesize Voice Converted feature to wav files
Vocoder=$(echo ${Vocoder} | tr '[A-Z]' '[a-z]')
echo "synthesize voice converted features to wav files using "${Vocoder}" vocoder..."

python ${MerlinDir}/misc/scripts/vocoder/${Vocoder}/synthesis.py ${MerlinDIr} ${rec_feat_dir} ${rec_wav_dir} $SamplingFreq ${scp_dir} 

echo "deleting intermediate synthesis files..."
./scripts/remove_intermediate_files.sh $global_config_file

echo "transformed audio files are in: "${rec_wav_dir}""

