#### basenet

Classes wrapping basic `pytorch` functionality.
  
  - Train for an epoch
  - Eval for an epoch
  - Predict
  - Learning rate schedules

##### Installation

```
conda create -n basenet_env python=3.5 pip -y
source activate basenet_env

pip install -r requirements.txt
conda install -y pytorch torchvision cuda90 -c pytorch
python setup.py install
```

#### Examples

```
cd examples
python cifar10.py
```