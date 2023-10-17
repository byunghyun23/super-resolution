# Super-Resolution Based on SRGAN with Sliding Window

## Introduction
This is a TensorFlow implementation for Super-Resolution based on SRGAN with Sliding Window.  
Ref: https://github.com/leftthomas/SRGAN

## Dataset
For training the model, you need to download [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).  
We used 16K images for trainand 300 images for validation.

## SRGAN + Sliding Window
SRGAN is affected by the upscaling factor.  
For example, if the size of the super-resolution image is 384\*384 and the upscaling factor is 4, the size of the input image must be 96\*96 to obtain good results.  
To achieve super-resolution regardless of the input image size, the following operations are performed.  
```
1. Divide the input image into patches.
2. Use each patch as input to the SRGAN model
3. Combine the output of SRGAN for each patch again.
```
![image](https://github.com/byunghyun23/super-resolution/blob/main/assets/fig1.png)

## Train
```
python train.py
```
After training, a super-resolution model and training process log are generated in the directory below.
```
--model
```

## Predict
You can get the super-resolution image by running
```
python predict.py --file_name file_name
```

## Demo
Also, you can also use the model using Gradio by running
```
python web.py
```
![image](https://github.com/byunghyun23/super-resolution/blob/main/assets/fig2.PNG)

