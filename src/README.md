# Depression Detect Workflow

Below is the basic workflow.

Gain access to the DAIC-WOZ data base and download the zip files to your project directory using:

```shell
wget -r -np -nH --cut-dirs=3 -R index.html --user=daicwozuser --ask-password  http://dcapswoz.ict.usc.edu/wwwdaicwoz/
```

### Data
1. run `extract_from_zip.py` to extract the wave files and transcript csv's from the zip files.

2. run `segmentation.py` to create a segmented wave files with silence and the virtual interview's speech removed. Feature extraction is performed on the segmented wave files.

### Features
3. `cnn_spectrograms` performs the short-time Fourier transform (STFT) into a spectrogram and and makes a matrix representation.

4. `spectrogram_dicts.py` builds dictionaries with keys of participant ids for the each class and values with the matrix representation of the spectrogram.

5. `random_sampling.py` is a script that returns a test/train split for input to the CNN based on random sampling from the spectorgram_dicts for each class. This was really important because of the class imbalance.

6. run `cnn.py` to train the Convolutional Neural Network (CNN).
