Machine-Learning Side-Channel Attacks on the GALACTICS Constant-Time Implementation of BLISS
===

## Overview

- [Prerequisites](#prerequisites)
  * [Python requirements](#pyreq)
  * [Attack 3 requirements](#att3req)
  * [Side-channel data for all three attacks](#scdata)
- [Running the attacks](#runatt)

## Prerequisites
### <a name="pyreq"></a> Python requirements
```shell script
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### <a name="att3req"></a> Attack 3 requirements
```shell script
sudo apt install -y libboost-all-dev libeigen3-dev libntl-dev libmpc-dev
```

### <a name="scdata"></a> Side-channel data for all three attacks
The dataset used for all three attack is publicly [available](https://zenodo.org/record/5101343/files/galactics_attack_data.7z?download=1).

Prerequisites:
```shell script
sudo apt-get install p7zip
```
Extract the data:
```shell script
7z x galactics_attack_data.7z
```

## <a name="runatt"></a> Running the attacks
The three attacks are divided into three different python notebooks. Each of them includes the full attack using only the dataset from Zenodo.

Start python notebook:
```shell script
python3 -m notebook
```

The notebook for Attack1 generates the models using the training data (data from device A). Therefore the notebook for Attack1 has to run through before attempting Attack2&3.
