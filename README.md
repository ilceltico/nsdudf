# Neural Surface Detection for Unsigned Distance Fields

[Project page](https://ilceltico.github.io/nsdudf/) | [ArXiv](https://arxiv.org/abs/2407.18381)

Pytorch implementation of the ECCV 2024 paper "Neural Surface Detection for Unsigned Distance Fields", Federico Stella, Nicolas Talabot, Hieu Le, Pascal Fua. École Polytechnique Fédérale de Lausanne (EPFL), Switzerland.

## Installation
### Step 1
For the code to work, you need:
```
Python
Numpy
Pytorch
Trimesh
Cython
Libigl
Setuptools
```

Alternatively, you can find my Conda environment for macOS in `requirements_macos_conda.txt`, and my Pip package list for CUDA in `requirements_cuda_pip.txt`. A GPU is not required to run the code, but of course it speeds up the execution.

### Step 2
Compile the Cython implementation of Marching Cubes.
```
cd custom_mc
export CFLAGS="-I path_to_numpy/core/include/ $CFLAGS"
python setup.py build_ext --inplace
```


## Usage
A usage example is provided in the file `example_extract_mesh.py`, with hopefully enough comments to understand everything. Let me know if clarifications are needed!

You can load our pre-trained weights, the same as in the paper with:
```
model = utils.load_model("model.pt", device)
```
The Pseudo-SDF is computed by calling
```
pseudo_sdf = compute_pseudo_sdf(model, lambda query_points: udf_and_grad_f(query_points, object), n_grid_samples=resolution)
```
with the required resolution. `udf_and_grad_f` is a function you should define, and for which you can find examples in the same file, which computes the UDF and the gradients for a specific object (neural or not).

The output can be meshed using Marching Cubes, which we provide with a slight interface modification to accept our Pseudo-SDF input.
```
mesh = mesh_marching_cubes(pseudo_sdf)
```

Training code and integration with DualMesh-UDF coming soon.

## Bibtex
If you find this work useful, please cite us!
```
@misc{stella2024neuralsurfacedetectionunsigned,
      title={Neural Surface Detection for Unsigned Distance Fields}, 
      author={Federico Stella and Nicolas Talabot and Hieu Le and Pascal Fua},
      year={2024},
      eprint={2407.18381},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.18381}, 
}
```