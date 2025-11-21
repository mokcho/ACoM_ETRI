# Audio Coding for Machines
ETRI project led by Jinju Kim, Seungho Kwon

Question : How can we adapt Audio Codecs for machine perception, rather than human perception for machine tasks?

## To-Do's
Methods
- [x] implement naiive filtering layers
- [x] implement frame-wise ROI pre-filtering
- [x] implement frequency-wise ROI pre-filtering
- [ ] save best-mask results as axis, percentage, region (frames, freqs)
- [ ] train lightweight neural network for prediction
      
Downstream Tasks
- [x] ESC-50
- [x] Sound Event Detection


## Dataset

1. Environment Sound Classification Dataset (ESC-50)

    To train/eval on ESC-50 prepare your dataset from [ESC-50 Official repo](https://github.com/karolpiczak/ESC-50)


2. Sound Event Detection Dataset (DESED)

    To train/eval on DESED prepare your dataset from [DCASE 2024 Official Website](https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels)


## Training


1. Environment Sound Classification (ESC-50)

 - before running the code, configure your config file for training. 
 - fold can be set from [1, 2, 3, 4, 5]
 - bitrate can be set from [1.5, 3, 6, 12, 24]

    ```bash
    python beats_trainer.py --configs $YOUR_CONFIG$ --fold $YOUR_FOLD$ --bitrate $YOUR_BITRATE$
    ```

2. Sound Event Detection (DESED)

    T.B.D


