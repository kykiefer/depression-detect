import os
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
import wave


"""
A script that iterates through the extracted wav files and uses
pyAudioAnalysis' silence extraction module to make a wav file containing the
segmented audio (when the participant is speaking -- silence and virtual
interviewer speech removed)
"""


def remove_silence(filename, out_dir, smoothing=1.0, weight=0.3, plot=False):
    """
    A function that implements pyAudioAnalysis' silence extraction module
    and creates wav files of the participant specific portions of audio. The
    smoothing and weight parameters were tuned for the AVEC 2016 dataset.

    Parameters
    ----------
    filename : filepath
        path to the input wav file
    out_dir : filepath
        path to the desired directory (where a participant folder will
        be created containing a 'PXXX_no_silence.wav' file)
    smoothing : float
        tunable parameter to compensate for sparseness of recordings
    weight : float
        probability threshold for silence removal used in SVM
    plot : bool
        plots SVM probabilities of silence (used in tuning)

    Returns
    -------
    A folder for each participant containing a single wav file
    (named 'PXXX_no_silence.wav') with the vast majority of silence
    and virtual interviewer speech removed. Feature extraction is
    performed on these segmented wav files.
    """
    partic_id = 'P' + filename.split('/')[-1].split('_')[0]  # PXXX
    if is_segmentable(partic_id):
        # create participant directory for segmented wav files
        participant_dir = os.path.join(out_dir, partic_id)
        if not os.path.exists(participant_dir):
            os.makedirs(participant_dir)

        os.chdir(participant_dir)

        [Fs, x] = aIO.readAudioFile(filename)
        segments = aS.silenceRemoval(x, Fs, 0.020, 0.020,
                                     smoothWindow=smoothing,
                                     Weight=weight,
                                     plot=plot)

        for s in segments:
            seg_name = "{:s}_{:.2f}-{:.2f}.wav".format(partic_id, s[0], s[1])
            wavfile.write(seg_name, Fs, x[int(Fs * s[0]):int(Fs * s[1])])

        # concatenate segmented wave files within participant directory
        concatenate_segments(participant_dir, partic_id)


def is_segmentable(partic_id):
    """
    A function that returns True if the participant's interview clip is not
    in the manually identified set of troubled clips. The clips below were
    not segmentable do to excessive static, proximity to the virtual
    interviewer, volume levels, etc.
    """
    troubled = set(['P300', 'P305', 'P306', 'P308', 'P315', 'P316', 'P343',
                    'P354', 'P362', 'P375', 'P378', 'P381', 'P382', 'P385',
                    'P387', 'P388', 'P390', 'P392', 'P393', 'P395', 'P408',
                    'P413', 'P421', 'P438', 'P473', 'P476', 'P479', 'P490',
                    'P492'])
    return partic_id not in troubled


def concatenate_segments(participant_dir, partic_id, remove_segment=True):
    """
    A function that concatenates all the wave files in a participants
    directory in to single wav file (with silence and other speakers removed)
    and writes in to the participant's directory, then removes the individual
    segments (when remove_segment=True).
    """
    infiles = os.listdir(participant_dir)  # list of wav files in directory
    outfile = '{}_no_silence.wav'.format(partic_id)

    data = []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
        if remove_segment:
            os.remove(infile)

    output = wave.open(outfile, 'wb')
    # details of the files must be the same (channel, frame rates, etc.)
    output.setparams(data[0][0])

    # write each segment to output
    for idx in range(len(data)):
        output.writeframes(data[idx][1])
    output.close()


if __name__ == '__main__':
    # directory containing raw wav files
    dir_name = '../../data/raw/audio'

    # directory where a participant folder will be created containing their
    # segmented wav file
    out_dir = '../../data/interim'

    # iterate through wav files in dir_name and create a segmented wav_file
    for file in os.listdir(dir_name):
        if file.endswith('.wav'):
            filename = os.path.join(dir_name, file)
            remove_silence(filename, out_dir)
