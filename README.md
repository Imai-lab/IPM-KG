# IPM-KG

This repository provides supplementary material for the following article:

> Tatsuya Tanaka, Toshiaki Katayama, and Takeshi Imai.  
> **Predicting the effects of drugs and unveiling their mechanisms of action using an interpretable pharmacodynamic mechanism knowledge graph (IPM-KG).**  
> *Computers in Biology and Medicine*, 184, 109419, 2025.  
> https://doi.org/10.1016/j.compbiomed.2024.109419

## Overview

IPM-KG is a research repository for missing predicate prediction and drug effect prediction in knowledge graphs using Graph Neural Networks.

The code in this repository was used as supplementary material for the above article. It includes implementations and related scripts for the experiments described in the paper.

This code is based in part on the following previous methods:

- E-GraphSAGE: https://doi.org/10.1109/NOMS54207.2022.9789878
- KG-Predict: https://www.sciencedirect.com/science/article/pii/S1532046422001496

## Repository structure

- `EdgeLabel_classification/`  
  Code related to missing predicate prediction / edge label classification.

- `DrugEffect_prediction/`  
  Code related to drug effect prediction.

- `requirements.txt`  
  Python package requirements.

## Citation

If you use this repository, please cite the following article:

```bibtex
@article{tanaka2025ipmkg,
  title = {Predicting the effects of drugs and unveiling their mechanisms of action using an interpretable pharmacodynamic mechanism knowledge graph (IPM-KG)},
  author = {Tanaka, Tatsuya and Katayama, Toshiaki and Imai, Takeshi},
  journal = {Computers in Biology and Medicine},
  volume = {184},
  pages = {109419},
  year = {2025},
  doi = {10.1016/j.compbiomed.2024.109419}
}
```

## Notes

This repository is provided as supplementary material for the associated paper and is intended to support reproducibility of the computational experiments described therein.
