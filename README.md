# Forest Fire Detection using Custom CNN and ATT Squeeze U-Net

## Overview
This project implements a forest fire detection system using the ATT Squeeze U-Net neural network architecture and custom CNN model which is chosen based on compartive results. The methodology is based on the paper by Zhang et al., and the dataset used is [RML2016](https://www.kaggle.com/datasets/momenhamdy/rml2016) from Kaggle.

## Dataset
The RML2016 dataset from Kaggle contains various labeled images for training and testing the forest fire detection model.

## Methodology of ATT Squeeze U-Net
- **ATT Squeeze U-Net**: Combines SqueezeNet and Attention U-Net for effective detection and segmentation.
  - **Modified SqueezeNet**: Uses depthwise separable convolutions and Channel Shuffle for efficient feature extraction.
  - **Attention Gates**: Highlight relevant features while suppressing irrelevant ones.
  - **DeFire Modules**: Custom modules for efficient up-sampling in the decoder.

## Requirements
- Python 3.7+
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
