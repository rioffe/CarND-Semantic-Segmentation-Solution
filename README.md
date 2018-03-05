# CarND-Semantic-Segmentation-Solution
Solution to Term 3, Project 2 of Udacity Nanodegree: Semantic Segmentation

To run the project:
```
    python main.py EPOCHS BATCH_SIZE LEARNING_RATE BETA
``` 

I experimented with EPOCHS of size 30, 50, 90, 120 (the last two only for batch sizes
16 and 20) and BATCH_SIZES of 4, 5, 8, 10, 12, 16, and 20, LEARNING_RATES of 0.001,
0.0003, 0.0001, and BETAS of 0.0001, 0.0003, 0.001, 0.003. 

In general, I found the best results were obtained for small batch sizes (4, 5) and typically
the number of epochs was 50 or smaller. 

I also experimented with Tensorflow 1.5 on NVidia GT1060 graphics card and Tensorflow 1.2.1
on AWS g3.4xlarge instance, which has NVidia Tesla M60 GPU.

The best results for both images and video were obtained with the following:
```
    python main.py 30 5 0.001 0.003
```  
Here are all the resulting segmented images (note, that you can also examine individual images
in runs directory):

![Segmented Images](runs/1519272238.3490055/animation.gif)

Losses over the Number of Images processed graph:

![Loss Graph](losses_vs_images_epochs_30_batchsz_5_lr_0.001_beta_0.003.png)

See the segmented video here (the best result was obtained on Tensorflow 1.2.1):

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/85jKOR98PYA/0.jpg)](https://youtu.be/85jKOR98PYA)

I attempted image augmentation by randomly darkening the input images and, even though
the image segmentation impoved on the images, the quality of the image segmentation on
the video decreased, so after much experimentation I had to abandon it.

