from sys import stderr
import os

import numpy as np

from pydub import AudioSegment
import soundfile
import librosa


# -> np.array (mono audio data), int (sampling rate)
def load_audio(fname, normalize=False):
    print("Loading file {}...".format(fname), end="", file=stderr)
    af = AudioSegment.from_file(fname)
    data = np.array([chan.get_array_of_samples()
                     for chan in af.split_to_mono()],
                    dtype=np.float32)
    rate = af.frame_rate
    channels = data.shape[0]

    if channels > 1:
        data = np.mean(data, axis=0)
    else:
        data = data[0]

    if normalize:
        data = data / np.amax(data)

    print("OK", file=stderr)
    return data, rate


# audios: [(data, rate), ...]
# -> [data, ...], rate
def resample_to_common(audios):
    datas, rates = zip(*audios)
    rate = min(rates)
    return (librosa.resample(data, orig, rate)
            for data, orig in zip(datas, rates)), rate


def export_audio(fname, data, rate):
    try:
        soundfile.write(fname, np.asarray(data, np.int16), rate)
    except TypeError:
        File = os.path.splitext(fname)
        fname = File[0] + ".wav"
        soundfile.write(fname, np.asarray(data, np.int16), rate)

        if File[1] == ".mp3":
            fnamemp3 = File[0] + ".mp3"
            AudioSegment.from_wav(fname).export(fnamemp3, format="mp3")
            os.remove(fname)
