# HerbMet: A Framework for Chinese Herbal Medicine Identification via Deep Learning and Metabolomics Data

## Introduction

Chinese herbal medicine has been used for thousands of years to prevent and treat diseases. Several high-value Chinese herbal medicines are used in drugs and foods that have proven medical and health benefits. Identifying species might be challenging because several herbal medicines contain similar bioactive constituents. Therefore, accurate identification of Chinese herbal medicines is crucial due to varying medicinal effects. This study aims to develop a high-performance artificial intelligence system to identify Chinese herbal medicines, especially to identify those from different species of the same genus. We summarize the contributions of our paper as follows:

1. We developed HerbMet, an efficient Chinese herbal medicine identification system, with a simple 1D-ResNet architecture and metabolomics data. The proposed method can provide a strong baseline for metabolomics data analysis.

2. The DDR is a regularization method based on standard Dropout technology. It can solve the problem of overfitting in the Chinese herbal medicine dataset and optimize algorithm performance.

3. We conduct extensive experiments on two Chinese herbal medicine metabolomics datasets, and the results demonstrate that our proposed method has significant advantages in terms of accuracy, robustness, and model effectiveness


## Proposed Method

This article presents HerbMet, an AI-based system for accurately identifying Chinese herbal medicines. To achieve this goal, we design 1D-ResNet architecture to extract discriminated features from the input samples and then employ Multilayer Perceptron (MLP) to map these representations to the final results. To alleviate overfitting, we also introduce the Double Dropout Regularization Module (DDR) to optimize the model performance. In order to evaluate the model performance, we conduct extensive experiments on our collected dataset, including the *Panax ginseng* and *Gleditsia sinensis* datasets.

<img src="images/fig_abs_01.png" width="100%">


## Environment

- The code is developed using python 3.10 on Ubuntu 20.04.
- The proposed method should be trained and validated in GPU.
- If you want to train the proposed segmentation method on custom dataset, Nvidia GPUs are needed. 
- This code is development and tested using one Nvidia GTX 4090 GPU.  


## Quick start

### Model training and validation

- Training:
    ```
    python main.py
    ```

**Notable:** For privacy and security reasons, the dataset cannot be upload to GitHub or Google Drive. If you are interseted in our project or dataset, please contact with us. 
