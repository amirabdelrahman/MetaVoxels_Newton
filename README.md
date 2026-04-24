# MetaVoxels Newton



<!-- MetaVoxels Designer re-written to be simulated using newton physics.


## Getting started
1. clone [https://github.com/newton-physics/newton](https://github.com/newton-physics/newton) and follow instructions on the repo for installation
2. clone this git repo to `/newton/newton/examples`
3. change ```/newton/newton/examples/__init__.py``` line 360 to: 

```python    
modules = ["basic", "cloth", "diffsim", "ik", "mpm", "robot", "selection", "sensors","metavoxels_newton"]
```
4. to run the wip examples

```bash
uv run -m newton.examples metavoxels
```

## RL workflow

1. add dependencies

```bash
uv add sb3-contrib gymnasium plotly tqdm rich
```

2. run RL

```bash
cd ./RL
uv run python ./run1.py
``` -->