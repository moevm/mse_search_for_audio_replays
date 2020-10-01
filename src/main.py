import argparse

def detect_reps(fnames):
    print("Requested repetitions detection for files: {}".format(
        ", ".join(fnames)))
    print("(not implemented yet)")

def denoise(fnames):
    print("Requested noise reduction for files: {}".format(
        ", ".join(fnames)))
    print("(not implemented yet)")

def main(argv):
    p = argparse.ArgumentParser(
        description="""Detect repetitions in audio tracks. In noise
        reduction mode, apply noise reduction to all specified
        files.""")
    p.add_argument("files", metavar="FILE", type=str, nargs="*",
                   help="""audio files to process (stdin if none
                   specified)""")
    p.add_argument("-d", "--denoise", action="store_true",
                   help="noise reduction mode")
    p.add_argument("-V", "--version", action="version", version="0.0.1")
    args = p.parse_args()

    if not args.files:
        args.files.append('-')

    if args.denoise:
        return denoise(args.files)
    else:
        return detect_reps(args.files)

if __name__ == "__main__":
    from sys import argv
    exit(main(argv) or 0)
