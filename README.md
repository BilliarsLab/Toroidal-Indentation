# Toroidal-Indentation
**For Measuring Anisotropy of Biomaterials Using Indentation Method**

This repository demonstrates the method detailed in our study, "Material Evaluation from Experimental Indentation Force," which provides an accessible tool for characterizing mechanical anisotropy of biological samples across scales. The approach combines toroidal indentation probes and neural networks for robust material property prediction. A demo and related files are available here: **[link to demo]**.

This page will be updated as needed.

## Authors:
For technical questions, contact:
- Juanyong Li: [jli16@wpi.edu](mailto:jli16@wpi.edu)

For collaboration inquiries, contact:
- Kristen Billiar: [kbilliar@wpi.edu](mailto:kbilliar@wpi.edu)

## Prerequisites
To replicate or extend the methodology, the following are required:
- Python 3.8+
- TensorFlow/Keras
- MATLAB 2022b (for preprocessing and basic simulations)
- Abaqus 2020 (for finite element analysis)
- Access to a 2PP 3D printer (e.g., Nanoscribe GT+) for fabricating toroidal probes

## Data Preprocessing
The experimental and simulated indentation data are normalized to allow comparison across probe sizes and indentation depths. The scripts for preprocessing include:
1. Normalizing force-indentation data based on indentation depth and probe radius.
2. Generating feature vectors from orthogonal force-displacement curves.

Detailed instructions for preprocessing are provided in `data_preprocessing/README.md`.

## Pretrained Neural Network Models
Navigate to `/pre-trained-models` to access the trained models:
- Three pretrained neural network models are provided based on finite element simulations (N=2208) using three distinct probe geometries:
  - R3.25
  - R7
  - R10

Each model predicts intrinsic anisotropic elastic moduli (E1 and E2) based on normalized force-indentation data.

## Acknowledgement
This project was supported by:
- NSF Grant CMMI 1761432
- ARMI BiofabUSA Grant T0137
- NIH Grant 1R15HL167235-01, R21NS136884
- Simulations performed using WPI's high-performance computing systems under NSF MRI Grant DMS-1337943.

## Citation
If you find this repository helpful in your work, please cite the associated paper:
> [Add citation details once published]

---

We welcome contributions and feedback from the community. For further information, refer to the accompanying documentation or contact the authors directly.
