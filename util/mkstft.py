#!/usr/bin/env python3

import argparse
import sys

import numpy as np
from pydub import AudioSegment
import librosa


def prinf(*args, **kwargs):
    return print(*args, file=sys.stderr, **kwargs)

def print_tf(tf, rate, file=sys.stdout):
    fprint = lambda *args, **kwargs: print(*args, file=file, **kwargs)
    fprint(tf.shape[1], end="")
    for i in range(tf.shape[1]):
        fprint(' {}'.format(librosa.frames_to_samples(i)/rate), end="")
    fprint()
    for i, freq in enumerate(librosa.fft_frequencies(sr=rate)):
        row = tf[i]
        fprint(freq, end="")
        for x in tf[i]:
            fprint(' {}'.format(np.abs(x)), end="")
        fprint()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('filename')
    p.add_argument('-t', '--time', metavar='T', type=float,
                   help='Only use first T seconds of audio')
    args = p.parse_args()

    af = AudioSegment.from_file(args.filename)
    data = np.array([chan.get_array_of_samples()
                     for chan in af.split_to_mono()])
    rate = af.frame_rate

    prinf(f"channels: {af.channels}")
    prinf(f"rate: {rate}")
    prinf(f"dtype: {data.dtype}")
    prinf(f"sample width: {af.sample_width}")
    prinf(f"shape: {data.shape}")
    prinf(f"length (by bitrate): {data.shape[1] / rate}")

    prinf(f"max: {np.amax(data)}")
    prinf(f"min: {np.amin(data)}")

    mono = (np.mean(data, axis=0)
            if data.shape[0] > 1
            else np.asarray(data[0], np.float32))

    prinf(f"mono shape: {mono.shape}")

    piece = (mono if args.time is None
             else mono[:round(rate * args.time)])

    tf = librosa.stft(piece);

    prinf(f"STFT matrix shape: {tf.shape}")

    print_tf(tf, rate)

    return 0

if __name__ == "__main__":
    exit(main())
