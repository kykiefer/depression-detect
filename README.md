# Detecting Depression From Acoustic Features

Depression detections...

## Table of Contents
1. [Dataset](#dataset)
2. [MRI Background](#mri-background)
    * [MRI Pre-Processing](#mri-pre-processing)
    * [Pulse Sequences](#pulse-sequences)
    * [Segmentation](#segmentation)
3. [High Grade Gliomas](#high-grade-gliomas)
4. [Convolutional Neural Networks](#convolutional-neural-networks)
    * [Model Architecture](#model-architecture)
    * [Training the Model](#training-the-model)  
    * [Patch Selection](#patch-selection)
    * [Results](#results)
5. [Future Directions](#future-directions)

## Dataset

All audio recordings and associated depression metrics were provided by the [DAIC-WOZ Database](http://dcapswoz.ict.usc.edu/), which was released as part of the 2016 Audio/Visual Emotional Challenge and Workshop [(AVEC 2016)](http://sspnet.eu/avec2016/). The dataset consists of 189 sessions, averaging 16 minutes, between a participant and virtual interviewer called Ellie, controlled by a human interviewer in another room. Prior to the interview, each participant completed a psychiatric questionnaire [(PHQ-8)](http://patienteducation.stanford.edu/research/phq.pdf) from which a binary classification for depression was derived as a truth label for each patient. A transcribed snippet is seen below:

> **Ellie** Who’s someone that’s been a positive influence in your life?

> **Participant** Uh my father.

> **Ellie** Can you tell me about that?

> **Participant** Yeah, he is a uh. He’s a very he’s a man of few words. And uh he's very calm. Slow to anger. And um very warm very loving man. Responsible. And uh he’s a gentleman has a great sense of style and he’s a great cook.

## MRI Background

Magnetic Resonance Imaging (MRI) is the most common diagnostic

### MRI pre-processing ([code](https://github.com/naldeborgh7575/brain_segmentation/blob/master/code/brain_pipeline.py))

One of the challenges in working with MRI data is dealing with

### Pulse sequences
There are multiple radio frequency pulse sequences that can be

### Segmentation
Notice now that a single patient will produce upwards of 600

## High Grade Gliomas

High-grade malignant brain tumors are generally associated with

## Convolutional Neural Networks

Convolutional Neural Networks(CNNs) are a powerful tool in the

### Model Architecture ([code](https://github.com/naldeborgh7575/brain_segmentation/blob/master/code/Segmentation_Models.py))

I use a four-layer Convolutional Neural Network (CNN) model


### Training the Model

I created the model using Keras and ran it on an Amazon AWS

### Patch Selection ([code](https://github.com/naldeborgh7575/brain_segmentation/blob/master/code/patch_library.py))
The purpose of training the model on patches (Figure 8) is to exploit the fact that a class of any given voxel is highly

### Results

Below is a summary of how well the current model is predicting. As more advances are made this section will be updated. A representative example of a tumor segmentation on test data is displayed in Figure 10. The model can identify each of the four

## Future Directions

While my model yields promising results, an application such as this leaves no room for errors or false positives. In a surgical setting it is essential to remove as much of the tumor mass as

## References

    1. Havaei, M. et. al, Brain Tumor Segmentation with Deep Neural Networks. arXiv preprint arXiv:1505.03540, 2015.
