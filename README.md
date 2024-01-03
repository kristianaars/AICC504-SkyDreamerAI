This project implements the [dreamerv3](https://danijar.com/project/dreamerv3/) algorithm to attempt to solve the Minecraft Skyblock challenge. It uses [minedojo](https://minedojo.org) as the environment supplier. 

To begin training run the file [train.py](./train.py). Tensorboard is also supported, please see log-output for tensorboard log path and run the following command.
```
tensorboard --logdir YOUR_LOGPATH_HERE
```

## Setup

### Before Starting
There is a bug in Gym that causes its installation to fail. This is fixed by running the following pip-update commands:
```

```

### JAX
Follow instructions at: https://github.com/google/jax#installation

For Ubuntu with CUDA-Cores, you may use as of 11/12/23:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Install requirements
```
pip install -r requirements.txt
```
