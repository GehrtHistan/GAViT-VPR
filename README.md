# GAViT-VPR

**Graph Attention-based Lightweight Re-ranking for Visual Place Recognition**

Official research implementation of **GAViT**, a lightweight graph attention-based re-ranking approach for Visual Place Recognition (VPR).

This repository accompanies the following research papers:

## Publications

### Conference Paper — MLMI 2026

*Improving Efficiency and Geometric Consistency in Transformer-based Visual Place Recognition*

Yuchen Guo, Masato Taki

2026 International Conference on Machine Learning and Machine Intelligence (MLMI 2026).

**Paper:** [ACM Digital Library](https://dl.acm.org/doi/10.1145/3841358.3841385)

**DOI:** 10.1145/3841358.3841385

### Extended Journal Paper — Under Review

An extended version of the MLMI 2026 paper has been submitted to *Transactions on Artificial Intelligence (TAI)* and is currently under review.

## Repository Contents

This repository contains two research components:

- **GAViT:** Graph Attention Network (GAT)-based lightweight re-ranking for Visual Place Recognition, aiming to reduce computational complexity while maintaining retrieval performance.
- **GeoConsistency:** Evaluation of geometric consistency in Vision Transformer backbones, including DeiT, Swin Transformer, DINOv2, and MAE.

## Pretrained Model Weights

Pretrained model weights for GAViT, including different configurations and ablation study variants, are hosted on Hugging Face.

**[Download GAViT Model Weights (Hugging Face)](https://huggingface.co/Gehrthistan33/GAViT-VPR/tree/main)**

The model repository contains `.pth` checkpoints for reproducibility and further research.

## Installation and Usage

GAViT is built upon the original [R2Former](https://github.com/bytedance/R2Former) framework. The overall environment setup, data preparation, training, and evaluation procedures follow the original R2Former implementation.

Please refer to the original R2Former repository for detailed instructions.

### 1. Download the Source Code

Clone the GAViT repository:

```bash
git clone https://github.com/GehrtHistan/GAViT-VPR.git
```

### 2. Select a GAViT Configuration

GAViT provides four configurations: `GAViT_basic_0`, `GAViT_basic_1`, `GAViT_basic_2`, and `GAViT_basic_3`.

To use a specific configuration, replace the original `R2Former/model/R2Former.py` file with the corresponding GAViT implementation.

| GAViT Configuration | Replacement File |
|---|---|
| GAViT_basic_0 | `R2Former_basic_0.py` |
| GAViT_basic_1 | `R2Former_basic_1.py` |
| GAViT_basic_2 | `R2Former_basic_2.py` |
| GAViT_basic_3 | `R2Former_basic_3.py` |

For example, to use `GAViT_basic_1`:

1. Locate `R2Former/model/R2Former.py` in the R2Former framework.
2. Back up the original `R2Former.py` file.
3. Copy `R2Former_basic_1.py` into the same directory.
4. Rename the copied file to `R2Former.py`, replacing the original implementation.

The resulting directory structure should be:

```text
R2Former/
└── model/
    └── R2Former.py  # Replaced with GAViT_basic_1
```

Repeat this procedure with the corresponding implementation file when switching between GAViT configurations.

### 3. Download Pretrained Model Weights

The pretrained model weights, including different GAViT configurations and ablation study variants, are available on Hugging Face:

**[GAViT-VPR Model Weights](https://huggingface.co/Gehrthistan33/GAViT-VPR/tree/main)**

Download the checkpoint (`.pth`) corresponding to the selected GAViT configuration.

**Important:** Ensure that the selected model implementation and checkpoint correspond to the same configuration. For ablation study models, use the matching architecture and checkpoint.

### 4. Training and Evaluation

After replacing the model implementation and selecting the corresponding checkpoint, follow the training and evaluation procedures described in the original R2Former repository.

The overall workflow remains consistent with R2Former, while the re-ranking architecture is modified according to the selected GAViT configuration.

Please refer to the original implementation for dataset preparation, configuration settings, and execution commands.

## Acknowledgements

This work builds upon R2Former:

[R2Former GitHub Repository](https://github.com/bytedance/R2Former)

## License

This repository is released under the Apache License 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
