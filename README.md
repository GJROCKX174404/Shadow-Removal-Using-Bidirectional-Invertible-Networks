# Shadow-Removal-Using-Bidirectional-Invertible-Network
## Overview
This Repository consists of a deep-learning model for Digital Image Shadow Removal using Color Transfer and Bidirectional Invertible Convoultions, developed using pytorch framework.
## Datasets
[SRD](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view?usp=drive_link)
[ISTD](https://drive.gURLoogle.com/file/d/1I0qw-65KBA6np8vIZzO6oeiOvcDBttAY/view?usp=drive_link)
[ISTD+](https://drive.google.com/file/d/1rsCSWrotVnKFUqu9A_Nw9Uf-bJq_ryOv/view?usp=drive_link)

## Experimental Setup
- **Framework:** PyTorch
- **GPU Hardware:** NVIDIA GeForce RTX 4090, 48GB (2 clusters)
- **CUDA Version:** 12.2
- **Data Augmentation:**
  - Random horizontal and vertical flipping
  - Random rotation by 180 degrees
- **Optimizer:** Adam
- **Initial Learning Rates:**
  - Color Transfer: $$4 \times 10^{-4}$$
  - MainNet: $$2 \times 10^{-4}$$







