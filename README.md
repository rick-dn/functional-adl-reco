# A Step Towards Automated Functional Assessment of Activities of Daily Living
[![DOI](https://img.shields.io/badge/DOI-10.1007/978--3--031--14771--5_13-blue)](https://doi.org/10.1007/978-3-031-14771-5_13)

![Alt text](https://github.com/rick-dn/functional-adl-reco/blob/main/functional_adl_dataset.webp)

***Summary***

This repository provides access to the Functional-ADL dataset and the code implementation of our pose-based two-stream multi-label activity recognition model presented in the paper "A Step Towards Automated Functional Assessment of Activities of Daily Living".

In the paper, we address the challenges in using current activity recognition systems for the functional assessment of physically impaired individuals. Existing datasets primarily focus on recognizing normal human activities and lack the nuances required to differentiate between how the same activity is performed by individuals with different physical impairments. This limitation hinders the development of automated systems for rehabilitation and assessment. To overcome this, we introduce the Functional-ADL dataset, a novel multi-label dataset that includes both normal and impairment-specific executions of common Activities of Daily Living (ADL). This dataset is designed to support the development of more sophisticated activity recognition models capable of discerning subtle differences in movement patterns caused by various physical impairments. In addition to the dataset, we propose a new pose-based two-stream multi-label activity recognition model. Our model employs a spatial stream to capture structural relationships between body joints and a temporal stream to encode the temporal dynamics of joint movements. The model also incorporates a Fisher Vector-based pooling mechanism to enhance feature representation.

We demonstrate that our proposed model outperforms existing state-of-the-art methods on the Functional-ADL dataset and a well-known ADL recognition dataset.

***Key Contributions***

* A novel multi-label functional ADL dataset (Functional-ADL) that includes normal and four different physical impairment-specific versions of ten common ADL.
* A pose-based two-stream functional ADL recognition model that integrates spatial-temporal body-pose encoding with Fisher Vector-based pooling.

***Important Notes:***

* This repository provides the code for reference only. Please raise an issue to access the dataset.
* The dataset can be used for research purposes only.

The research is a part of my PhD theis: "A Vision-Based Approach For Assisting Functional Assessment Involving Activities of Daily Living" which can be found at: https://research.edgehill.ac.uk/ws/portalfiles/portal/38954446/23744596_phd_thesis_24_05_09.pdf

![Alt text](https://github.com/rick-dn/functional-adl-reco/blob/main/functional_adl_dataset.png)

For more information on the dataset please refer to Chapter 5 of the thesis: "Functional Activity Recognition Dataset"

This is code repository is just for reference only. This repository in not maintained and we do not intend to address issues related to the code. However, please feel free to raise an issue if you want to more information on the dataset and it processing steps.
