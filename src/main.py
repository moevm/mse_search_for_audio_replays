#!/usr/bin/env python3

import argparse
from sys import stderr
import os
import math

import numpy as np
import librosa

from audio import load_audio, export_audio, resample_to_common
from progress import simple_progressbar
from repetitions import get_repetitions


def next_pow2(x):
    p2 = 1
    while p2 < x:
        p2 <<= 1
    return p2


def noise_spec(noise_data, rate, frame_length=0.05):
    frame_samples = round(frame_length * rate)
    n_fft = next_pow2(frame_samples)

    tf = librosa.stft(noise_data,
                      n_fft=n_fft,
                      win_length=frame_samples)

    return np.amax(np.abs(tf), axis=1)


def reduce_noise(data, rate, noise, frame_length=0.05, progress=None):
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

        if progress is not None:
            progress(len(dest) / len(tf))

    dest_arr = np.array(dest).T
    return librosa.istft(dest_arr,
                         win_length=frame_samples,
                         length=data.shape[0])


def detect_reps(fnames, **kwargs):
    def timestr(seconds_fp):
        mseconds = round(seconds_fp * 1e3)
        mseconds_only = mseconds % 1000
        seconds = mseconds // 1000
        seconds_only = seconds % 60
        minutes = seconds // 60
        minutes_only = minutes % 60
        hours = minutes // 60
        return "{:02d}:{:02d}:{:02d}.{:03d}".format(
            hours, minutes_only, seconds_only, mseconds_only)

    signals, rate = resample_to_common(
        load_audio(fname, normalize=True)
        for fname in fnames
    )

    with simple_progressbar('Detecting repetitions') as bar:
        for t1, t2, l in get_repetitions(signals, rate,
                                         progress=bar.update,
                                         **kwargs):
            i1, tt1 = t1
            i2, tt2 = t2
            print("repetition: {} {}--{} <=> {} {}--{}"
                  .format(fnames[i1], timestr(tt1), timestr(tt1+l),
                          fnames[i2], timestr(tt2), timestr(tt2+l)))


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

        with simple_progressbar(fname) as bar:
            noise = noise_tab.get(src_rate)
            if noise is None:
                samp = librosa.resample(samp_data, samp_rate, src_rate)
                noise = noise_spec(samp, src_rate)
                noise_tab[src_rate] = noise

            res_data = reduce_noise(src_data, src_rate, noise, progress=bar.update)
        format_dot_place = fname.rfind(".", 0, len(fname))
        format_line = fname[format_dot_place + 1:] if (len(fname) > format_dot_place + 1) else ""
        if backup_suffix:
            if format_dot_place == -1:
                os.rename(fname, fname + backup_suffix)
            else:
                os.rename(fname, fname[:format_dot_place] + backup_suffix + '.' + format_line)
        export_audio(fname, res_data, src_rate)

    return 0


def main_denoise(args):
    return denoise(args.sample, args.backup_suffix,
                   args.files)


def main_reps(args):
    return detect_reps(args.files,
                       frame_length=args.frame_length,
                       threshold_k=args.threshold,
                       window_size=args.window_length,
                       merge_distance_threshold=args.merge_threshold,
                       min_final_length=args.min_length,
                       val_threshold_k=args.window_threshold,
                       max_near_distance=args.parallel_merge_threshold)


def main(argv):
    p = argparse.ArgumentParser(
        description='Performs noise reduction and searches for repetitions in audio files.')

    p.add_argument("-V", "--version", action="version",
                   version="0.0.1")

    sp = p.add_subparsers()

    p_reps = sp.add_parser(
        "repetitions",
        help="detect repetitions in specified file")

    p_reps.add_argument(
        "files", metavar="FILE", type=str, nargs="+",
        help="audio files to detect repetitions in")
    p_reps.add_argument(
        "-l", "--min-length", metavar="SEC", type=float, default=2.0,
        help="minimal length of a repetition in seconds")
    p_reps.add_argument(
        "-f", "--frame-length", metavar="SEC", type=float, default=0.05,
        help="length of STFT frame in seconds")
    p_reps.add_argument(
        "-t", "--threshold", metavar="K", type=float, default=3,
        help="comparison threshold (no dimension; lower for stricter comparisons)")
    p_reps.add_argument(
        "--window-length", metavar="SEC", type=float, default=0.5,
        help="length of comparison window in seconds")
    p_reps.add_argument(
        "--merge-threshold", metavar="SEC", type=float, default=0.5,
        help="maximum distance between matches in seconds to merge them")
    p_reps.add_argument(
        "--window-threshold", metavar="K", type=float, default=1.5,
        help="comparison threshold for whole window (no dimension; lower for stricter comparisons)")
    p_reps.add_argument(
        "--parallel-merge-threshold", metavar="SEC", type=float, default=0.5,
        help="distance between similar matches' frames in seconds to merge them")
    p_reps.set_defaults(func=main_reps)

    p_denoise = sp.add_parser(
        "denoise",
        help="apply noise reduction to all specified files")

    p_denoise.add_argument(
        "sample", metavar="SAMPLE",
        help="noise sample")
    p_denoise.add_argument(
        "files", metavar="FILE", type=str, nargs="+",
        help="audio files to process")
    p_denoise.add_argument(
        "-b", "--backup-suffix", metavar="SUFFIX",
        help="if set, copy source files before overwriting")
    p_denoise.set_defaults(func=main_denoise)

    args = p.parse_args()

    if 'func' not in args:
        p.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    from sys import argv

    exit(main(argv) or 0)
