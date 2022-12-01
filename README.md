# Live Reconstruction using Wigner Distribution Deconvolution
This repository contains a reformulation of Wigner Distribution Deconvolution (WDD) for live reconstruction and scripts
to reproduce the results in the related paper.

To benchmark against an existing implementation of WDD we refer to the implementation in pyPtychoSTEM that can be acessed at the following link: https://gitlab.com/pyptychostem/pyptychostem

The datasets for graphene and WSe2 can be downloaded here:
1. Graphene: https://zenodo.org/record/4476506#.YxioTdVBxH5
2. WSe2: https://zenodo.org/record/6477629#.YxioMdVBxH5
3. Experimental SrTiO3: https://zenodo.org/record/5113449#.Yxnh7dVBxH4

## Requirements

The requirements for live reconstruction can be installed as dependencies with ```pip install .``` from this repository.
The scripts and notebooks require the following additional packages:

```
click
https://github.com/sk1p/python-perf-utils
psutil
tikzplotlib
```


This repository is presented to reproduce the result and act as supplementary material for the paper below. For continued development please refer the Ptychography 4.0 repository: https://github.com/Ptychography-4-0/ptychography

## Citation

## License

See file LICENSE

