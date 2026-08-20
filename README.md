# dens-city: High-Performance Molecular Density Functional Theory & Neural Operator Platform

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://python.org)
[![CUDA: 12.0+](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PufferLib: Zero-Copy C](https://img.shields.io/badge/PufferLib-Zero--Copy%20C-orange.svg)](https://github.com/PufferAI/PufferLib)

> **Note**: This project is undergoing a clean rebuild from scratch.

## 1. FreeSolv Database & Molecular Topologies
- **Submodule**: [FreeSolv](https://github.com/MobleyLab/FreeSolv) (Mobley Lab hydration database)
- **Test Data**: Canonical Tripos `.mol2` structure files and AMBER GAFF parameters located in [`test_data/`](./test_data)
- **Data Extractor**: `python scripts/init_data.py` extracts the raw GAFF `.mol2` archives into `data/`
- **cDFT Solute Validation**: `python scripts/generate_cdft_input_validation.py` compiles the complete solute input table into `data/input_validation.txt`

## 2. License
GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
