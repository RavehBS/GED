#!/bin/bash
cd $HHC_HOME
mkdir data
for dataset in zoo iris glass ; do
  mkdir data/$dataset
  wget -P data/$dataset https://archive.ics.uci.edu/ml/machine-learning-databases/$dataset/$dataset.data
  wget -P data/$dataset https://archive.ics.uci.edu/ml/machine-learning-databases/$dataset/$dataset.names
done

mkdir data/breast_cancer
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JiVoATmiWRqUN_TuZdzPxkxkb69V38qp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JiVoATmiWRqUN_TuZdzPxkxkb69V38qp" -O data/breast_cancer/data_mrna.txt && rm -rf /tmp/cookies.txt