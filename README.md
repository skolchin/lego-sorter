# Lego Sorter project

(c) lego-sorter team (kol+chem), 2022-2023

LegoSorter is an open-source project aiming to build a machine
which would be able to recognize and, eventually, automatically sort
various Lego pieces.

This repo contains a software part of the machine, hardware setup
is performed behind the scenes.

The project uses TensorFlow Google's library to perform
Lego pieces image classification. Currently, it employs
pre-made VGG-16 CNN trained in transfer learning mode on
custom rendered Lego pieces image dataset.

So far the project is in some king of R&D stage so no timings are set.
We wil definetelly share the results to the public as soon as
we got the result :)

## Setup

Basic Python setup is required, see requirements.txt for list of packages.

For training, Tensorflow's GPU version is recommended, see, for example,
this [guide](https://towardsdatascience.com/how-to-finally-install-tensorflow-gpu-on-windows-10-63527910f255)
on how to install it. For NVIDIA cards
it should be working fine with CUDA 11.2.0_460.89 and CUDNN 11.2_8.1.1.33
(available at
[gdrive](https://drive.google.com/drive/folders/1OgHnA7X_Ey-GSy8eepNUFhnVercTeq_e?usp=share_link)).

Pre-trained model checkpoints are also available at the link above.
They should be placed under `checkpoints` directory.

Lego images dataset is quite big and I chose not to publish it.

Currently it was tested under Windows only, Linux support would come up later.

## Model training

To perform model training, run `train.py` script. Several image augmentation modes
are supported:

    -g, --gray  Convert images to grayscale
    -e, --edges Convert images to some kind of wireframe by detecting edges
        and eliminating all other data
    -x, --emboss    Combine wireframes with actual images thus highligthing
        the edges (could be combined with --gray)
    -z, --zoom      Apply zoom augmentation while training (this theoretically
        would allow to capture right features at different scales, but
        it would severely slows down the training)

Model weights for the options are saved to different directories under `checkpoints` root.

Other options:

    -n, --epoch Number of epoch to train for (normally its 50-100)
    --noshow    By default the script shows samples and learning results,
        this option turns it off
    --nosave    Turns off model saving (usefull for quick test)

## Model testing

Trained model can be tested with `test.py` script. It supports all augmentation modes
listed above and the following options:

    -c, --label     Show random prediction samples for given label (class name)
    -f, --files     Run prediction over given files (could be specified more than once)
    -m, --confusion Build and show a confusion matrix

If no options provided, the script would show some random prediction samples
from images dataset.

## Inferrence

The `static_pipe.py` script is a prototype of the project's control panel. When run,
it will connect to a video camera and allow to classify Lego pieces put under
that camera.

All image augmentation modes are supported as script's options.
When started, several hot keys are available:

    B   Capture a static background (this must be done in order to detect new pieces)
    D   Show debug information (additional windows and log)
    P   Display captured image as it was preprocessed by the model
    C   Start or stop saving video stream to a file (in `out` directory`)
    S   Display video camera settings and adjustments
    Q or ESC    Quit

## Controller state machine
![alt text](https://github.com/skolchin/lego-sorter/tree/controller_debug/states.jpg?raw=true)