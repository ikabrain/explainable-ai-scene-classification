# Explainable AI: Scene Classification and GradCam Visualization

[![Pixi Badge](https://img.shields.io/badge/PIXI-f9c405)](https://pixi.sh/)
![Python Badge](https://img.shields.io/badge/python-3.10.6-blue?logo=python)
[![Tensorflow Badge](https://img.shields.io/badge/tensorflow-2.17.0-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras Badge](https://img.shields.io/badge/keras%20_3-D00000?logo=keras)](https://keras.io/)


This project involves training a deep learning model to predict the type of scenery in images. In addition, we are going to use a technique known as Grad-Cam to help explain how AI models think. This could be practically used for detecting the type of scenery from the satellite images.

## Dataset Description
<img src="figures/class_distribution.png" alt="Dataset Structure" height=300/>

## Model Architecture - ResNet (Deep CNN with Residual Blocks)
<img src="figures/conv_identity_block.png" alt="Conv Identity Block" width=700/>
<img src="figures/res-block.png" alt="Residual Block" width=700/>

*(Refer to [this image](figures/resnet18_model.png) to see model implementation)*

## Model Performance
<img src="figures/confusion_matrix.png" alt="Confusion Matrix" height=500/>

## Testing input/output
<img src="figures/sample_predictions.png" alt="Sample Predictions"/>

## Grad-CAM Visualization
<img src="figures/gradcam_visualizations.png" alt="Grad-CAM Visualization" width=700/>

## References

Ahmed, R. (n.d.). Explainable AI: Scene Classification and GradCam Visualization [MOOC]. Coursera. https://www.coursera.org/projects/scene-classification-gradcam

Duong, B. T. (2021). Explainable AI: Scene classification and Grad-CAM visualization [Source code]. GitHub. https://github.com/baotramduong/Explainable-AI-Scene-Classification-and-GradCam-Visualization

TensorFlow. (n.d.). TensorFlow documentation. https://www.tensorflow.org/

Keras. (n.d.). Keras documentation. https://keras.io/

Prefix.dev. (n.d.). Pixi documentation. https://pixi.prefix.dev/latest/
