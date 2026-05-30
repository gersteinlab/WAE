# WAE: Neural Topic Modeling with Wasserstein Autoencoders in Large Scale Multiomic Datasets

This repository implements a Neural Topic Modeling framework using Wasserstein Autoencoders (WAE) designed specifically for processing and analyzing large-scale  Host Transcriptome and Microbiome . It includes the Wasserstein Topic Model (WTM), multi-modality processing, baseline comparative models (like mLDA), and post-analysis workflows.

---

## 📂 Repository Structure

* **`WAE/`**: Core source code containing the neural network architectures, Wasserstein loss functions, and data loading pipelines for the WTM model, which  We adapted an existing WAE implementation from https://github.com/zll17/Neural_Topic_Models
* **`mLDA/`**: Contains code for the comparative multi-modal Latent Dirichlet Allocation baseline model.
* **`sample_data/`**: Directory containing sample datasets (e.g., COVID multiomic data profiles) to test and run the models.
* **`run_wtm_demo.sh`**: A helper bash script to easily configure and launch the WTM pipeline with varying hyperparameters.
* **Analysis Notebooks (`*.ipynb`)**: Jupyter notebooks dedicated to downstream analysis, visualization, and validation of the learned latent topics.

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure you have Python 3.8+ and PyTorch installed. Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/gersteinlab/WAE.git](https://github.com/gersteinlab/WAE.git)
cd WAE
bash run_wtm_demo.sh <beta> <epochs> <alpha>

