#!/usr/bin/env bash

python train.py -C yaml/cfg_pretrain_2021.yaml
python train.py -C yaml/cfg_pseudo_75.yaml
python train.py -C yaml/cfg_pseudo_104.yaml
python train.py -C yaml/cfg_pseudo_104_xlarge.yaml
python train.py -C yaml/cfg_pseudo_140.yaml

python train.py -C yaml/pretrain-type-v9.yaml
python train.py -C yaml/pseudo-pretrain-75-seedblend-aux-v10-ff.yaml
python train.py -C yaml/pseudo-pretrain-104-seedblend-aux-v10-ff.yaml

python train.py -C yaml/shrewd-rook-3ep-ff.yaml
python train.py -C yaml/big-ocelot-ff.yaml
python train.py -C yaml/meteoric-bettong-ff.yaml
python train.py -C yaml/saffron-rook-ff.yaml
python train.py -C yaml/conscious-uakari-ff.yaml
python train.py -C yaml/olivine-spaniel-ff.yaml

python train.py -C yaml/awesome-rose-ff.yaml
python train.py -C yaml/axiomatic-vulture-ff.yaml
python train.py -C yaml/funky-funk-ff.yaml
python train.py -C yaml/honest-apple-ff.yaml
python train.py -C yaml/lame-flame-ff.yaml
python train.py -C yaml/pastel-frog-ff.yaml
python train.py -C yaml/smart-bumblebee-ff.yaml
python train.py -C yaml/valiant-degu-ff.yaml
