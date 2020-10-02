#!/usr/bin/env python3

import argparse
from sys import stderr

def detect_reps(fnames):
    print("Requested repetitions detection for files: {}".format(
        ", ".join(fnames)))
    print("(not implemented yet)")

def denoise(sample_fname, fnames):
    if sample_fname is None:
        print("error: noise sample not specified for denoise mode.",
              file=stderr)
        return 1
    print("Requested noise reduction for files: {}".format(
        ", ".join(fnames)))
    print("Sample: {}".format(sample_fname))
    print("(not implemented yet)")

def main(argv):
    p = argparse.ArgumentParser(
        description="""Detect repetitions in audio tracks. In noise
        reduction mode, apply noise reduction to all specified
        files.""")
    p.add_argument("files", metavar="FILE", type=str, nargs="+",
                   help="audio files to process")
    p.add_argument("-d", "--denoise", action="store_true",
                   help="noise reduction mode")
    p.add_argument("-s", "--noise-sample", metavar="SAMPLE",
                   help="noise sample (required with --denoise)")
    p.add_argument("-V", "--version", action="version", version="0.0.1")
    args = p.parse_args()

    if args.denoise:
        return denoise(args.noise_sample, args.files)
    else:
        return detect_reps(args.files)

if __name__ == "__main__":
    from sys import argv
    exit(main(argv) or 0)
