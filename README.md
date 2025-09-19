# Audio Coding for Machines
ETRI project led by Jinju Kim, Seungho Kwon

## To-Do's
Methods
- [x] implement naiive filtering layers
- [ ] implement frame-wise ROI pre-filtering
- [ ] implement frequency-wise ROI pre-filtering
      
Downstream Tasks
- [x] ESC-50
- [x] Sound Event Detection


## Dataset

1. Environment Sound Classification Dataset (ESC-50)


2. Sound Event Detection Dataset (DESEC)


## Training


1. Environment Sound Classification

 - before running the code, configure your config file for training. 
 - fold can be set from [1, 2, 3, 4, 5]
 - bitrate can be set from [1.5, 3, 6, 12, 24]

python beats_trainer.py --configs $YOUR_CONFIG$ --fold $YOUR_FOLD$ --bitrate $YOUR_BITRATE$


