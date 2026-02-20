# MeMo-Hyfydy-Workshop

## Setting up the environment
```
conda create --name hfd_env python=3.9
conda activate hfd_env
pip install -r requirements.txt
cd sconegym_main
pip install -e .
cd ..
```

## Testing the environment
```
python test_env.py
```

## Training model
```
python train.py
```

## Generating rollout
```
python rollout.py [filename]
```