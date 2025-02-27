# CNN Architecture Experiment: Optimising a DenseNet-Based Model

### Overview
This project explores the process of improving a **Convolutional Neural Network (CNN)** based on a **DenseNet** architecture, focusing on incremental optimizations while working within limited computational resources. The primary objective was to determine how much improvement could be achieved by systematically refining a relatively simple model without access to large-scale hardware.

### Approach
The experiment followed an iterative approach, where various modifications were applied and evaluated across 10 experiments. Key areas of focus included:

- Data Preprocessing & Augmentation – Evaluating how different preprocessing techniques affect model performance.

- Hyperparameter Tuning – Adjusting learning rates, batch sizes, and optimiser choices.

- Network Modifications – Experimenting with layer configurations, dropout rates, and activation functions.

- Attention Layers - Incorporating advanced layer architectures to older CNN models.

- Regularisation Techniques – Implementing weight decay and dropout to prevent overfitting.

- Model Evaluation & Visualisation – Tracking accuracy, loss curves, and confusion matrices to assess improvements. 

### Challenged & Constraints

Due to limited computational resources, the focus was on:
- ✔ Efficient model architecture modifications rather than excessive parameter scaling.
- ✔ Running models with optimised batch sizes to balance memory usage and training speed.
- ✔ Prioritising practical, resource-aware improvements over brute-force experimentation.

### Key Results

- Performance Gains: Achieved measurable accuracy improvements through targeted optimisations.
- Efficiency Trade-offs: Certain modifications required balancing between accuracy gains and computational cost.
- Insights on DenseNet Variants: Highlighted the most effective tweaks for improving a lightweight DenseNet-based CNN under constraints.