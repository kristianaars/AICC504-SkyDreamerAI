## Setup

###
There is a bug in Gym that causes its installation to fail. This is fixed by running the following pip-update commands:
```

```

### JAX
Follow instructions at: https://github.com/google/jax#installation

For Ubuntu with CUDA-Cores, you may use as of 11/12/23:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

```

```

###

### Dreamerv3
```
git clone https://github.com/danijar/dreamerv3
mv dreamerv3 .dreamerv3
mv .dreamerv3/dreamerv3 ./dreamerv3
rm -rf .dreamerv3
```

### Minedojo
