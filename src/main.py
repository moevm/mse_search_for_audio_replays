#!/usr/bin/env python3

import argparse
from sys import stderr
import os
import math

import numpy as np
from progressbar import ProgressBar, Percentage, Bar, ETA
from pydub import AudioSegment
import librosa
import soundfile


def next_pow2(x):
    p2 = 1
    while p2 < x:
        p2 <<= 1
    return p2


# -> np.array (mono audio data), int (sampling rate)
def load_audio(fname):
    print("Loading file {}...".format(fname), end="")
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

    print("OK")
    return data, rate


def noise_spec(noise_data, rate, frame_length=0.05):
    frame_samples = round(frame_length * rate)
    n_fft = next_pow2(frame_samples)

    tf = librosa.stft(noise_data,
                      n_fft=n_fft,
                      win_length=frame_samples)

    return np.amax(np.abs(tf), axis=1)


def reduce_noise(data, rate, noise, frame_length=0.05):
    widgets = ['Test: ', Percentage(), ' ',
               Bar(marker='#', left='[', right=']'),
               ' ', ETA()]
    p_bar = ProgressBar(widgets=widgets, maxval=100).start()
    counter = 0

    frame_samples = round(frame_length * rate)
    n_fft = next_pow2(frame_samples)
    tf = librosa.stft(data,
                      n_fft=n_fft,
                      win_length=frame_samples).T

    def f(fa, na, p):
        afa = np.abs(fa)
        if afa > na and p > na:
            return fa
        else:
            return 0

    dest = []

    N = 8
    prev = np.zeros(tf.shape[1], dtype=np.float32)
    prevs = [None] * N
    i = 0

    for frame in tf:
        row = [f(fa, na, p)
               for fa, na, p in zip(frame, noise, prev / N)]

        curr = np.abs(frame)
        if prevs[i] is not None:
            prev -= prevs[i]
        prevs[i] = curr
        prev += curr
        i = (i + 1) % N

        dest.append(row)

        counter = counter + 1
        p_bar.update(counter / len(tf) * 100)

    p_bar.finish()

    dest_arr = np.array(dest).T
    return librosa.istft(dest_arr,
                         win_length=frame_samples,
                         length=data.shape[0])


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
    noise_tab = {samp_rate: noise_spec(samp_data, samp_rate)}

    for fname in fnames:
        try:
            src_data, src_rate = load_audio(fname)
        except FileNotFoundError:
            print("Error File {} not found!".format(fname), file=stderr)
            continue

        noise = noise_tab.get(src_rate)
        if noise is None:
            samp = librosa.resample(samp_data, samp_rate, src_rate)
            noise = noise_spec(samp, src_rate)
            noise_tab[src_rate] = noise

        print("Reducing noise from {}".format(fname))
        res_data = reduce_noise(src_data, src_rate, noise)
        format_dot_place = fname.rfind(".", 0, len(fname))
        format_line = fname[format_dot_place + 1:] if (len(fname) > format_dot_place + 1) else ""
        if backup_suffix:
            if format_dot_place == -1:
                os.rename(fname, fname + backup_suffix)
            else:
                os.rename(fname, fname[:format_dot_place] + backup_suffix + '.' + format_line)
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
