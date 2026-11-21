# GAViT-VPR

**Graph Attention-based Lightweight Re-ranking for Visual Place Recognition**

Official research implementation of **GAViT**, a lightweight graph attention-based re-ranking approach for Visual Place Recognition (VPR).

This repository accompanies the following research papers:

### Publications

**Conference Paper — MLMI 2026**

*Improving Efficiency and Geometric Consistency in Transformer-based Visual Place Recognition*

Yuchen Guo, Masato Taki

2026 International Conference on Machine Learning and Machine Intelligence (MLMI 2026).

**Paper:** [ACM Digital Library](https://dl.acm.org/doi/10.1145/3841358.3841385)

**DOI:** 10.1145/3841358.3841385

**Extended Journal Paper — Under Review**

An extended version of the MLMI 2026 paper has been submitted to *Transactions on Artificial Intelligence (TAI)* and is currently under review.

### Repository Contents

This repository contains two research components:

- **GAViT:** Graph Attention Network (GAT)-based lightweight re-ranking for Visual Place Recognition, aiming to reduce computational complexity while maintaining retrieval performance.
- **GeoConsistency:** Evaluation of geometric consistency in Vision Transformer backbones, including DeiT, Swin Transformer, DINOv2, and MAE.

### Pretrained Model Weights

Pretrained model weights for GAViT, including different configurations and ablation study variants, are hosted on Hugging Face.

**[Download GAViT Model Weights (Hugging Face)](https://huggingface.co/Gehrthistan33/GAViT-VPR/tree/main)**

The model repository contains `.pth` checkpoints for reproducibility and further research.

### Acknowledgements

This work builds upon R2Former:

[R2Former GitHub Repository](https://github.com/bytedance/R2Former)

### License

This repository is released under the Apache License 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
