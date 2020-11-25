#!/usr/bin/env python3

import argparse
from sys import stderr
import os

import numpy as np
from progressbar import ProgressBar, Percentage, Bar, ETA
from pydub import AudioSegment
import librosa
import soundfile

from repetitions import get_repetitions


# -> np.array (mono audio data), int (sampling rate)
def load_audio(fname, normalize=False):
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

    if normalize:
        data = data / np.amax(data)

    print("OK")
    return data, rate


def noise_spec(noise_data):
    tf = librosa.stft(noise_data)
    return np.amax(np.abs(tf), axis=1)


def reduce_noise(data, noise):
    widgets = ['Test: ', Percentage(), ' ',
               Bar(marker='#', left='[', right=']'),
               ' ', ETA()]
    p_bar = ProgressBar(widgets=widgets, maxval=100).start()
    counter = 0
    length = data.shape[0]
    tf = librosa.stft(data).T
    dest = []
    for frame in tf:
        dest.append([fa if np.abs(fa) > na else fa * 0.1
                     for na, fa in zip(noise, frame)])
        counter = counter + 1
        p_bar.update(counter / len(tf) * 100)
    p_bar.finish()
    dest = np.array(dest)
    return librosa.istft(dest.T, length=length)


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

    if len(fnames) != 1:
        print("Cannot detect repetitions across files yet")
        return 1
    fname = fnames[0]
    data, rate = load_audio(fname, normalize=True)

    for t1, t2, l in get_repetitions(data, rate, **kwargs):
        print("repetition: {}--{} <=> {}--{}".format(timestr(t1),
                                                     timestr(t1+l),
                                                     timestr(t2),
                                                     timestr(t2+l)))


def denoise(sample_fname, backup_suffix, fnames):
    if sample_fname is None:
        print("error: noise sample not specified for denoise mode.",
              file=stderr)
        return 1

    samp_data, samp_rate = load_audio(sample_fname)
    noise = noise_spec(samp_data)

    for fname in fnames:
        try:
            src_data, src_rate = load_audio(fname)
        except FileNotFoundError:
            print("Error File {} not found!".format(fname), file=stderr)
            continue
        print("Reducing noise from {}".format(fname))
        res_data = reduce_noise(src_data, noise)
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
    return detect_reps([args.file],
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
        "file", metavar="FILE", type=str,
        help="audio file to detect repetitions in")
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
