# Live Reconstruction using Wigner Distribution Deconvolution
This repository contains a reformulation of Wigner Distribution Deconvolution (WDD) for live reconstruction and scripts
to reproduce the results in the related paper.

To benchmark against an existing implementation of WDD we refer to the implementation in pyPtychoSTEM that can be acessed at the following link: https://gitlab.com/pyptychostem/pyptychostem

The datasets for graphene and SrTiO3 can be downloaded here:
1. Graphene: https://zenodo.org/record/4476506#.YxioTdVBxH5
2. Experimental SrTiO3: https://zenodo.org/record/5113449#.Yxnh7dVBxH4

The result to reproduce the images and graphs can be downloaded here:

Result: https://zenodo.org/record/7390871#.Y4oe7zPMJH5

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
```
@article{bangun2022wdd,
  title={Wigner Distribution Deconvolution Adaptation for Live Ptychography
    Reconstruction},
  author={Bangun, Arya and Baumeister, Paul F and Clausen, Alex and Weber, Dieter, Dunin-Borkowski, Rafal E.},
  journal={arXiv preprint (WILL BE UPDATED)},
  year={2022}
            }
```
## License


See file LICENSE

