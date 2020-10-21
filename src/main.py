#!/usr/bin/env python3

import argparse
from sys import stderr
import os

import numpy as np
from pydub import AudioSegment
import librosa
import soundfile

# -> np.array (mono audio data), int (sampling rate)
def load_audio(fname):
    af = AudioSegment.from_file(fname)
    data = np.array([chan.get_array_of_samples()
                     for chan in af.split_to_mono()])
    rate = af.frame_rate
    channels = data.shape[0]

    if channels > 1:
        data = np.mean(data, axis=0)

    return data, rate

def noise_spec(noise_data):
    tf = librosa.stft(noise_data)
    av = np.mean(np.abs(tf), axis=1)
    mx = np.amax(av)
    norm = av / mx
    return norm**2

def reduce_noise(data, noise):
    length = data.shape[0]
    tf = librosa.stft(data).T
    dest = tf * (1 - noise)
    return librosa.istft(dest.T, length=length)

def export_audio(fname, data, rate):
    soundfile.write(fname, np.asarray(data, np.int16), rate)



def detect_reps(fnames):
    print("Requested repetitions detection for files: {}".format(
        ", ".join(fnames)))
    print("(not implemented yet)")

def denoise(sample_fname, backup_suffix, fnames):
    if sample_fname is None:
        print("error: noise sample not specified for denoise mode.",
              file=stderr)
        return 1

    samp_data, samp_rate = load_audio(sample_fname)
    noise = noise_spec(samp_data)

    for fname in fnames:
        src_data, src_rate = load_audio(fname)
        res_data = reduce_noise(src_data, noise)
        if backup_suffix:
            os.rename(fname, fname + backup_suffix)
        export_audio(fname, res_data, src_rate)

    return 0

def main(argv):
    p = argparse.ArgumentParser(
        description="""Detect repetitions in audio tracks. In noise
        reduction mode, apply noise reduction to all specified
        files.""")
    p.add_argument("files", metavar="FILE", type=str, nargs="+",
                   help="audio files to process")
    p.add_argument("-d", "--denoise", action="store_true",
                   help="noise reduction mode")
    p.add_argument("-b", "--backup-suffix", metavar="SUFFIX",
                   help="if set, copy source files before overwriting")
    p.add_argument("-s", "--noise-sample", metavar="SAMPLE",
                   help="noise sample (required with --denoise)")
    p.add_argument("-V", "--version", action="version", version="0.0.1")
    args = p.parse_args()

    if args.denoise:
        return denoise(args.noise_sample, args.backup_suffix,
                       args.files)
    else:
        return detect_reps(args.files)

if __name__ == "__main__":
    from sys import argv
    exit(main(argv) or 0)
