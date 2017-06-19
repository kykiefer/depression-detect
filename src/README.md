# Depression Detect Workflow
Below is the basic workflow I performed. I'm running in my Python 2.7.13 environment on macOS.

Gain access to the [DAIC-WOZ database](http://dcapswoz.ict.usc.edu/) and download the zip files to your project directory by running the following command in your shell:

```shell
wget -r -np -nH --cut-dirs=3 -R index.html --user=daicwozuser --ask-password  http://dcapswoz.ict.usc.edu/wwwdaicwoz/
```

### Folder Structure
```
src
└───README.md   
└───data
│   ├──extract_from_zip.py
│   ├──segmentation.py
│
└───features
│   ├──spectrograms.py
│   ├──dev_data.py
│   ├──spectrogram_dicts.py
│   ├──random_sampling.py
│   ├──cnn.py
│   ├──plot_metrics.py
│
└───models
    ├──README.md
```

### Data
1. Run `extract_from_zip.py` to extract the wav files of the interviews and interview transcription csv files from the zip files.

2. Run `segmentation.py` to create segmented wav files for each participant (silence and the virtual interviewer's speech removed). Feature extraction is performed on the segmented wav files.

### Features
3. `spectrograms.py` performs the short-time Fourier transform ([STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)) on the segmented wav files, transforming the wav files into a matrix representation of a spectrogram. The vertical axis representing frequency, the horizontal axis representing time, and a value in the matrix representing the intensity (in decibels) of the frequency component at a particular time.

4. `dev_data.py` creates a dataframe including participant depression labels used in model development.

5. `spectrogram_dicts.py` builds dictionaries with keys of participant IDs for the each class and values with the matrix representation of the entire segmented wav file's spectrogram.

6. `random_sampling.py` returns 40 random, 4 second spectrograms for each participant. Then, participants from each class are randomly selected in equal proportion as input to the Convolutional Neural Network (CNN). This was critical step in reducing model bias.

7. `cnn.py` performs normalization on the spectrogram and preps the images for Keras. Then trains and evaluates the network.

8. `plot_metrics.py` plots loss, accuracy and ROC curve.

### Models
9. `cnn_final.h5` is a file containing the configuration and weights of the trained convolutional neural network. The file was too large for GitHub, so there exists a `README.md` as a placeholder with instruction on how to access it.
