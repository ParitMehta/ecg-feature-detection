#!/bin/bash
mkdir -p data/ptbxl
wget -r -N -c -np -nH --cut-dirs=3 -P data/ptbxl \
    https://physionet.org/files/ptb-xl/1.0.3/
