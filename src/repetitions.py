#!/usr/bin/env python3

import math
import numpy as np
import sys
import librosa


def signal_windows_match_matrix_stft(signal, wsize, rate, hop_ratio=4):
    n_fft = 1 << math.ceil(math.log2(wsize))
    hop = wsize // hop_ratio
    tf = np.abs(librosa.stft(signal,
                             n_fft,
                             win_length=wsize,
                             hop_length=hop)).T

    tf /= np.amax(tf)

    n = tf.shape[0]
    mtx = np.zeros([n, n])

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((tf[i] - tf[j]) ** 2)
            mtx[i, j] = d

    times = [librosa.frames_to_samples(i, hop_length=hop) / rate
             for i in range(tf.shape[0])]
    return mtx, times


def walk_with_window(mtx, x, thresh, winlen, valthresh=math.inf):
    wnd = np.empty(winlen)
    n = mtx.shape[0]
    end = n - x

    thresh_scaled = thresh * winlen

    for i in range(winlen):
        wnd[i] = mtx[i, x+i]
    s = np.sum(wnd)

    n_large = sum(x > valthresh for x in wnd)

    idx = 0
    for i in range(winlen, end):
        if s <= thresh_scaled and n_large == 0:
            yield i - winlen
        s -= wnd[idx]
        if wnd[idx] > valthresh:
            n_large -= 1
        wnd[idx] = mtx[i, x+i]
        s += wnd[idx]
        if wnd[idx] > valthresh:
            n_large += 1
        idx = 0 if idx == winlen - 1 else (idx + 1)

    if s <= thresh_scaled and n_large == 0:
        yield end - winlen


def buffer_matches(it, dist_thresh):
    acc_start = acc_len = None
    for i in it:
        if (acc_start is not None
            and i - acc_start <= acc_len + dist_thresh):
            acc_len = i - acc_start
        else:
            if acc_start is not None:
                yield acc_start, acc_len
            acc_len = 0
            acc_start = i
    if acc_start is not None:
        yield acc_start, acc_len


def find_matches(mtx, thresh, win_length, dist_thresh, val_thresh):
    n = mtx.shape[0]
    for x in range(win_length, n - win_length + 1):
        for i, l in buffer_matches(
                walk_with_window(mtx, x, thresh,
                                 win_length, val_thresh),
                dist_thresh):
            yield i, x+i, l + win_length


def filter_by_length(it, min_length):
    for t1, t2, l in it:
        if l >= min_length:
            yield t1, t2, l


def filter_self_overlapping(it):
    for t1, t2, l in it:
        if t1 + l <= t2:
            yield t1, t2, l


def filter_near(it, max_distance):
    def lines_near(ln1, ln2):
        y1, x1, l1 = ln1
        y2, x2, l2 = ln2
        dx = abs((x1 - y1) - (x2 - y2))
        if dx > max_distance:
            return False

        bmax = max(y1, y2)
        emin = min(y1+l1, y2+l2)
        common = emin - bmax
        if common < min(l1, l2) / 2:
            return False

        return True

    def result_lines(bins):
        for b in bins:
            bmin = min(ln[0] for ln in b)
            emax = max(ln[0]+ln[2] for ln in b)
            y, x, l = max(b, key=lambda ln: ln[2])
            db = y - bmin
            de = emax - (y + l)
            yield y-db, x-db, l+db+de

    have_merges = False
    bins = []
    for line in it:
        ch_bs = []
        for b in bins:
            for l in b:
                if lines_near(line, l):
                    ch_bs.append(b)
                    have_merges = True
                    break
        nbin = [line]
        for ch_b in ch_bs:
            bins.remove(ch_b)
            nbin.extend(ch_b)
        bins.append(nbin)

    # if have_merges:
        # yield from filter_near(result_lines(bins), max_distance)
    # else:
    yield from result_lines(bins)


def filter_close(it, epsilon):
    yield from it
    return

    eps2 = epsilon * epsilon

    def pt_close(x1, y1, x2, y2):
        return (x2-x1)**2+(y2-y1)**2 < eps2

    def pts_close(ln1, ln2):
        t11, t12, l1 = ln1
        t21, t22, l2 = ln2
        return (pt_close(t11, t12, t21, t22)
                and pt_close(t11+l1, t12+l1, t21+l2, t22+l2))

    lines = []
    for line in it:
        if not any(pts_close(ln, line) for ln in lines):
            lines.append(line)
            yield line


def print_gpmatrix(mtx, times=None, file=sys.stdout):
    fprint = lambda *args, **kwargs: print(*args, file=file, **kwargs)
    ts = times.__getitem__ if times is not None else lambda i: i

    fprint(mtx.shape[1], end="")
    for j in range(mtx.shape[1]):
        fprint(' {}'.format(ts(j)), end="")
    fprint()
    for i in range(mtx.shape[0]):
        row = mtx[i]
        fprint(ts(i), end="")
        for x in mtx[i]:
            fprint(' {}'.format(x), end="")
        fprint()


if __name__ == "__main__":
    from pydub import AudioSegment

    filename, s_frame_sec, s_k = sys.argv[1:]

    af = AudioSegment.from_file(filename)
    data = np.array([chan.get_array_of_samples()
                     for chan in af.split_to_mono()])
    rate = af.frame_rate
    samples = lambda sec: round(sec * rate)

    mono = (np.mean(data, axis=0)
                    if data.shape[0] > 1
            else np.asarray(data[0], np.float32))

    norm = mono / np.amax(np.abs(mono))

    frame_sec = float(s_frame_sec)
    frame_sz = samples(frame_sec)

    HOP_RATIO = 4

    mtx, times = signal_windows_match_matrix_stft(
        norm, frame_sz, rate, HOP_RATIO)

    frames = lambda sec: round(sec / frame_sec * HOP_RATIO)
    seconds = lambda idx: times[idx]

    print_gpmatrix(mtx, times)

    WIN_SEC = 0.5               # window size
    DIST_SEC = 0.5              # merge distance
    FILTER_LENGTH = 2           # minimal repetition length

    k = float(s_k)
    thresh = 62.5e-6 * frame_sz * k

    print('#', 'time1', 'time2', 'length', file=sys.stdout)
    for mt1, mt2, mlen in \
        filter_near(
            filter_self_overlapping(
                filter_by_length(
                    find_matches(
                        mtx, thresh,
                        frames(WIN_SEC),
                        frames(DIST_SEC),
                        thresh * 1.5),
                    frames(FILTER_LENGTH))),
            frames(0.5)):
        print(seconds(mt1), seconds(mt2), seconds(mlen),
              file=sys.stdout)
