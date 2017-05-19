# Depression Detect Workflow
Below is the basic workflow I performed. Many of the scripts use my specific file paths and will have to be altered to run on your machine. I'm running on macOS in my Python 2.7.13 environment.

For dependencies see: [requirements.txt]()

Gain access to the [DAIC-WOZ database](http://dcapswoz.ict.usc.edu/) and download the zip files to your project directory by running the following command in your shell:

```shell
wget -r -np -nH --cut-dirs=3 -R index.html --user=daicwozuser --ask-password  http://dcapswoz.ict.usc.edu/wwwdaicwoz/
```

### Data
1. Run `extract_from_zip.py` to extract the wav files of the interviews and interview transcription csv files from the zip files.

2. Run `segmentation.py` to create segmented wav files for each participant (silence and the virtual interview's speech removed). Feature extraction is performed on the segmented wav files.

### Features
3. `spectrograms.py` performs the short-time Fourier transform ([STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)) on the segmented wav files into a spectrogram matrix. Rows are frequency bins, columns are time, and the values in the matrix are the intensity levels (IN DECIBELS???)

4. `spectrogram_dicts.py` builds dictionaries with keys of participant ids for the each class and values with the matrix representation of the spectrogram.

5. `random_sampling.py` is a script that returns a test/train split for input to the Convolutional Neural Network (CNN) based on random sampling from the `spectrogram_dicts` for each class. This was really important because of the class imbalance.

6. Run `cnn.py` to train the CNN.
