# Audio Coding for Machines
ETRI project by Jinju Kim (project lead), Seungho Kwon

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

To prepare datasets, simply use the following script.

    ```bash
    ./scripts/dataset.sh
    ```

1. Environment Sound Classification Dataset (ESC-50)

    Prepare your dataset manually from [ESC-50 Official repo](https://github.com/karolpiczak/ESC-50)


2. Sound Event Detection Dataset (DESED)

    Prepare your dataset manually from [DCASE 2024 Official Website](https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels)

## Codecs (WIP)

1. EnCodec

2. Opus 

    ```bash
    apt-get update && apt-get install -y libopus0 libopus-dev
    ```

## Audio Models (WIP)

You can download both models using following script. Follow manual download below if you encounter any problems. If all fails - you could use huggingface instead of a downloaded checkpoint.

    ```bash
    ./scripts/audio.sh
    ```

1. BEATs

    The original github repository provides multiple checkpoints of pretrained & finetuned versions of BEATs model.

2. AST



## Training


1. Environment Sound Classification (ESC-50)

 - before running the code, configure your config file for training. 
 - fold can be set from [1, 2, 3, 4, 5]
 - bitrate can be set from [1.5, 3, 6, 12, 24]

    ```bash
    python beats_trainer.py --configs $YOUR_CONFIG$ --fold $YOUR_FOLD$ --bitrate $YOUR_BITRATE$
    ```

- Training a Classifier

currently runs with AST

    ```bash
    scripts/cls_train.sh
    ```

2. Sound Event Detection (DESED)

    T.B.D


