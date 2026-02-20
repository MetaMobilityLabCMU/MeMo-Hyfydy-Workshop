# MeMo-Hyfydy-Workshop

## Setting up the environment
Run these commands in terminal:
```
conda create --name hfd_env python=3.9
conda activate hfd_env
pip install -r requirements.txt
cd sconegym_main
pip install -e .
cd ..
```
And try running:
```
python train.py
```

## Training model
```
python train.py
```

## Generating rollout
```
python rollout.py [filename]
```