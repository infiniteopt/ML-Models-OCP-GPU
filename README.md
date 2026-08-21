# Embedded Machine Learning Models in InfiniteOpt problems on GPU
This repository contains the source code used for the embedded ML model case studies 
presented in the thesis "GPU-Accelerated Infinite-Dimensional Optimization" by Evelyn Gondosiswanto. Note that running these on GPU require an NVIDIA GPU with CUDA support.

## Building the repository
Clone the repository and navigate to its root directory:
```bash
git clone https://github.com/infiniteopt/ML-Models-OCP-GPU
cd ML-Models-OCP-GPU
```

## Setting up the Python Environment
Python is required to run the case studies using the PyTorch models. Once Python is installed, create and activate a virtual environment, then install the required packages:
```bash
python -m venv pytorch-env
source pytorch-env/bin/activate
python -m pip install -r requirements.txt
```

## Setting up the Julia Environment
You'll need to install Julia (available at https://julialang.org/downloads/). Then before starting the `REPL`, configure `PythonCall` to use the Python environment you just created:
```bash
export JULIA_PYTHONCALL_EXE="$(pwd)/pytorch-env/bin/python"
```

Then start Julia using the repository's project environment:
```bash
julia --project=.
```

From the Julia `REPL`, instantiate the dependencies specified by `Project.toml` and `Manifest.toml`:
```julia
julia> ]

(ML-Models-OCP-GPU) pkg> instantiate
```

Note that the manifest includes the path to a local `HSL_jll` file, which allows access to the HSL solver `ma97` required for reproducing these results. Ensure that you download `HSL_jll` from the HSL website and that the manifest points to it before running the benchmarks:
```julia
julia> ]
# Add or dev your HSL_jll file
]resolve
```

From here, both your Julia and Python environments are set up and you can run the benchmarks accordingly.