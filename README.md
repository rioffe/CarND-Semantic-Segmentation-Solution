# CarND-Semantic-Segmentation-Solution
Solution to Term 3, Project 2 of Udacity Nanodegree: Semantic Segmentation

To run the project:
```
    python main.py EPOCHS BATCH_SIZE LEARNING_RATE BETA
``` 

The best results for both images and video were obtained with the following:
```
    python main.py 30 5 0.001 0.003
```  

![Segmented Images](runs/1519272238.3490055/animation.gif)

![Loss Graph](losses_vs_images_epochs_30_batchsz_5_lr_0.001_beta_0.003.png)

See video here:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/85jKOR98PYA/0.jpg)](https://youtu.be/85jKOR98PYA)

