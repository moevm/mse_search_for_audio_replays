import math
from itertools import chain

import numpy as np
import sys
import librosa

from progress import subprogress


# note: wsize and hop are in samples
def signal_stft(signal, wsize, hop):
    n_fft = 1 << math.ceil(math.log2(wsize))
    tf = np.abs(librosa.stft(signal,
                             n_fft,
                             win_length=wsize,
                             hop_length=hop)).T

    tf /= np.amax(tf)

    return tf


def stft_to_distmtx(tf, progress=None):
    n = tf.shape[0]
    mtx = np.zeros([n, n])

    c = 0
    total = n * n // 2

    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((tf[i] - tf[j]) ** 2)
            mtx[i, j] = d
            if progress is not None:
                progress(c / total)
                c += 1

    return mtx


def walk_with_window(mtx, x, thresh, winlen, valthresh=0,
                     k_thresh=1.0,
                     progress=None):
    wnd = np.empty(winlen)
    n = mtx.shape[0]
    end = n - x

    for i in range(winlen):
        wnd[i] = mtx[i, x+i]
    s = np.sum(wnd)

    n_large = sum(x > valthresh for x in wnd)

    total = end - winlen

    idx = 0
    for i in range(winlen, end):
        k = s / winlen / thresh
        if k <= k_thresh and n_large == 0:
            yield i - winlen, k

        s -= wnd[idx]
        if wnd[idx] > valthresh:
            n_large -= 1
        wnd[idx] = mtx[i, x+i]
        s += wnd[idx]
        if wnd[idx] > valthresh:
            n_large += 1
        idx = 0 if idx == winlen - 1 else (idx + 1)

        if progress is not None:
            progress(i / total)

    k = s / winlen / thresh
    if k <= k_thresh and n_large == 0:
        yield end - winlen, k


def buffer_matches(it, dist_thresh):
    acc_start = acc_len = sd = None
    for i, d in it:
        if (acc_start is not None
            and i - acc_start + 1 <= acc_len + dist_thresh):
            acc_len = i - acc_start + 1
            sd += d
        else:
            if acc_start is not None:
                yield acc_start, acc_len, sd / acc_len
            acc_len = 1
            acc_start = i
            sd = d
    if acc_start is not None:
        yield acc_start, acc_len, sd / acc_len


def find_matches(mtx, thresh, win_length, dist_thresh, val_thresh,
                 progress=None):
    n = mtx.shape[0]

    c = 0
    total = n * n // 2

    high = n - win_length + 1
    for x in range(win_length, high):
        delta = n - x - win_length

        for i, l, d in buffer_matches(
                walk_with_window(
                    mtx, x, thresh,
                    win_length,
                    valthresh=val_thresh,
                    progress=subprogress(progress,
                                         c / total,
                                         delta / total)),
                dist_thresh):

            yield i, x+i, l + win_length, d

        c += delta


def mk_filter_by_length(min_length):
    def f(it):
        for t1, t2, l, d in it:
            if l >= min_length:
                yield t1, t2, l, d
    return f


def filter_self_overlapping(it):
    for t1, t2, l, d in it:
        if t1 + l <= t2:
            yield t1, t2, l, d


def merge_matches(ms):
    t1_min = min(t1 for t1, t2, l, d in ms)
    t2_min = min(t2 - t1 for t1, t2, l, d in ms)
    t1_max = max(t1 + l for t1, t2, l, d in ms)

    l = t1_max - t1_min

    l_total = sum(ln[2] for ln in ms)
    d = sum(d * l / l_total
            for t1, t2, l, d in ms)

    return t1_min, t1_min + t2_min, l, d


def mk_filter_near(max_distance):
    def f(it):
        def lines_near(ln1, ln2):
            y1, x1, l1, d1 = ln1
            y2, x2, l2, d2 = ln2
            dx = abs((x1 - y1) - (x2 - y2))
            if dx > max_distance:
                return False

            bmax = max(y1, y2)
            emin = min(y1+l1, y2+l2)
            common = emin - bmax
            if common < min(l1, l2) / 2:
                return False

            return True

        bins = []
        for line in it:
            ch_bs = []
            for b in bins:
                for l in b:
                    if lines_near(line, l):
                        ch_bs.append(b)
                        break
            nbin = [line]
            for ch_b in ch_bs:
                bins.remove(ch_b)
                nbin.extend(ch_b)
            bins.append(nbin)

        for b in bins:
            yield merge_matches(b)

    return f


def mk_filter_dist(max_d):
    def f(it):
        for t1, t2, l, d in it:
            if d <= max_d:
                yield t1, t2, l, d
    return f


def maybe_split(lengths, t1, t2, l):
    end = t1 + l

    dt = 0
    for dl in sorted(t0 - t1 for t0 in
                     chain(lengths, (l - t2 + t1 for l in lengths))
                     if t1 < t0 < end):
        if dl == dt:
            continue
        yield t1+dt, t2+dt, dl-dt
        dt = dl

    if dt < l:
        yield t1+dt, t2+dt, l-dt


def mk_filter_multitrack(lengths):
    def f(it):
        for t1, t2, l, d in it:
            yield from ((t1, t2, l, d)
                        for t1, t2, l
                        in maybe_split(lengths, t1, t2, l))
    return f


def mk_tfs_to_timestamp(lengths, seconds):
    def f(frames):
        for i, l in enumerate(lengths):
            if frames < l:
                return i, seconds(frames)
            else:
                frames -= l
    return f


def multifilter(source, *filters):
    for f in filters:
        source = f(source)
    return source


def get_repetitions(signals, rate,
                    frame_length=0.05,
                    threshold_k=3,
                    hop_ratio=4,
                    window_size=0.5,
                    merge_distance_threshold=0.5,
                    min_final_length=2,
                    val_threshold_k=1.5,
                    max_near_distance=0.5,
                    progress=None):

    frame_samps = round(frame_length * rate)
    hop_secs = frame_length / hop_ratio
    hop_samps = frame_samps // hop_ratio

    tfs = [signal_stft(s, frame_samps, hop_samps) for s in signals]
    tf = np.concatenate(tfs, axis=0)
    lengths = [tf.shape[0] for tf in tfs]
    mtx = stft_to_distmtx(tf, subprogress(progress, 0, 1/2))

    frames = lambda sec: round(sec / hop_secs)
    seconds = lambda frames: frames * hop_secs
    rev = mk_tfs_to_timestamp(lengths, seconds)

    # adjust the magic number together with threshold_k
    threshold = 62.5e-6 * frame_samps * threshold_k

    for mt1, mt2, mlen, mdist in multifilter(
            find_matches(
                mtx, threshold,
                frames(window_size),
                frames(merge_distance_threshold),
                threshold * val_threshold_k,
                subprogress(progress, 1/2, 1/2)),
            mk_filter_by_length(frames(min_final_length)),
            filter_self_overlapping,
            mk_filter_near(frames(max_near_distance)),
            mk_filter_dist(threshold),
            mk_filter_multitrack(lengths)):

        i1, t1 = rev(mt1)
        i2, t2 = rev(mt2)
        l = seconds(mlen)
        prob = (threshold - mdist) / threshold

        yield (i1, t1), (i2, t2), l, prob
