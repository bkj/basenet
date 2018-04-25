#### basenet

Classes wrapping basic `pytorch` functionality.
  
  - Train for an epoch
  - Eval for an epoch
  - Predict
  - Learning rate schedules

##### Installation

```
conda create -n basenet4_env python=3.6 pip -y
source activate basenet4_env

pip install -r requirements.txt
conda install -y pytorch torchvision cuda90 -c pytorch
python setup.py install
```

#### Examples

```
cd examples/cifar
./cifar10.sh
```