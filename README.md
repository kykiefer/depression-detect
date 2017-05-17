# Detecting Depression From Acoustic Features
A automated device for detecting depression from acoustic features in speech seeks to lower the barrier of entry of seeking help for potential mental illness and reinforce a medical professionals' diagnoses. Early detection and treatment of depression is essential in promoting remission, preventing relapse, and reducing the emotional burden of a disease. Current diagnoses are primarily subjective and early signs of depression are difficult for humans to quantify (i.e. changes in vocal inflection) but have potential to be quantified by machine learning algorithms in a wearable device.

## Table of Contents
1. [Dataset](#dataset)
2. [Acoustic Features of Depressed Speech](#acoustic-features-of-depressed-speech)
    * [Segmentation](#segmentation-code)
    * [Feature Extraction](#feature-extraction)
3. [Convolutional Neural Networks](#convolutional-neural-networks)
    * [Model Architecture](#model-architecture)
    * [Training the Model](#training-the-model)  
    * [Results](#results)
4. [Future Directions](#future-directions)

## Dataset
All audio recordings and associated depression metrics were provided by the [DAIC-WOZ Database](http://dcapswoz.ict.usc.edu/), which was released as part of the 2016 Audio/Visual Emotional Challenge and Workshop ([AVEC 2016](http://sspnet.eu/avec2016/)). The dataset consists of 189 sessions, averaging 16 minutes, between a participant and virtual interviewer called Ellie, controlled by a human interviewer in another room. Prior to the interview, each participant completed a psychiatric questionnaire ([PHQ-8](http://patienteducation.stanford.edu/research/phq.pdf)) from which a binary classification for depression was derived as a truth label for each patient. A transcribed snippet is seen below:

> **Ellie** Who’s someone that’s been a positive influence in your life?

> **Participant** Uh my father.

> **Ellie** Can you tell me about that?

> **Participant** Yeah, he is a uh. He’s a very he’s a man of few words. And uh he's very calm. Slow to anger. And um very warm very loving man. Responsible. And uh he’s a gentleman has a great sense of style and he’s a great cook.

<img alt="Virtual interview with Ellie" src="images/interview_with_ellie.png" width='400'>

<sub><b>Figure 1: </b> Virtual interview with Ellie. </sub>  

## Acoustic Features of Depressed Speech
While some research focuses on the semantic content of audio signals in predicting depression, I decided to focus on the [prosodic](https://en.wikipedia.org/wiki/Prosody_(linguistics) features. Things that are detectable to a listener of in terms of pitch, loudness, speaking rate, rhythm, voice quality, and articulation. (cite Automated Audiovisual Depression Analysis). Some features that have been found to be promising predictors of depression include using short sentences, flat intonation, fundamental frequency, Mel frequency cepstral coefficients ([MFCCs](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)).

### Segmentation ([code](https://github.com/kykiefer/depression-detect/blob/master/src/data/remove_silence.py))

The first step in being able to analyze a person's prosodic features of speech is being able to segment the person's speech from silence, other speakers, and noise. Fortunately, the participant's in the DAIC-WOZ study were wearing close proximity microphones and were in low noise environments, which allowed for fairly complete segmentation using [pyAudioAnanlysis' segmentation module](https://github.com/tyiannak/pyAudioAnalysis/wiki/5.-Segmentation). When implementing the algorithm in a wearable, [speaker diarisation](https://en.wikipedia.org/wiki/Speaker_diarisation) and background noise removal would obviously have to be extensively explored, but in interest of establishing a minimum viable product, extensive testing and tuning for segmentation robustness was forgone.

### Feature Extraction
There are several ways to approach this problem and this is by far the most critical component to building a successful model within this space. Approaches vary from extracting short-term and mid-term audio features such as MFCCs, [chroma vectors](https://en.wikipedia.org/wiki/Chroma_feature), [zero crossing rate](https://en.wikipedia.org/wiki/Zero-crossing_rate), etc. and feeding them as inputs to a Support Vector Machine (SVM). Since pyAudioAnalysis makes short-term feature extraction fairly streamlined, my first go at the problem was building a short-term feature matrix of 50ms audio segments of the 34 short-term features available form pyAudioAnalysis.

Results...

The state of the art emotion and language detection models seem to be using neural networks. So with limited initial success with an SVM, I wanted to focus my efforts on a convolutional neural network (CNN) and potentially combining it with Long Short Term Memory (LSTM). The most popular way to implement a CNN on audio signals is to provide it with a series of [spectrograms](https://en.wikipedia.org/wiki/Spectrogram).

<img alt="Spectrogram" src="images/spectrogram.png" width='700'>
<sub><b>Figure 2: </b> Spectrogram of man saying "Watching different TV shows". </sub>  

## Convolutional Neural Networks
Place holder

### Model Architecture
Place holder

### Training the Model
I created the model using Keras and ran it on an Amazon AWS

### Results
Below is a summary of how well the current model is predicting. As more advances are made this section will be updated. A representative example of a tumor segmentation on test data is displayed in Figure 10. The model can identify each of the four

## Future Directions
The model yields promising results...

I envision the model being implemented in a wearable device. The device prompts you to answer a simple question in the morning and a simple question before bed on a daily basis. The model stores your predicted depression score and tracks it over time such that the model can learn from your baseline (using a Bayesian approach). If a threshold is crossed, it notifies you to seek help or in extreme cases where action is not being taken, notifies an emergency contact to help you help yourself.

I'm currently excited about the results and and will be monitoring pull requests. However, accessing the DAIC-WOZ database requires signing an agreement form. Access can be granted [here](http://dcapswoz.ict.usc.edu/).

How I am prioritizing future efforts:
1. Segmentation robustness
2. Incorporate Vocal Tract Length Perturbation ([VTLP](http://www.cs.toronto.edu/~ndjaitly/jaitly-icml13.pdf))
3. Combine CNN with LSTM

## References

    1. Gratch J, Artstein R, Lucas GM, Stratou G, Scherer S, Nazarian A, Wood R, Boberg J, DeVault D, Marsella S, Traum DR. The Distress Analysis Interview Corpus of human and computer interviews. InLREC 2014 May (pp. 3123-3128)
    2. Placeholder
