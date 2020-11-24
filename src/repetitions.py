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

    for b in bins:
        bmin = min(ln[0] for ln in b)
        emax = max(ln[0]+ln[2] for ln in b)
        y, x, l = max(b, key=lambda ln: ln[2])
        db = y - bmin
        de = emax - (y + l)
        yield y-db, x-db, l+db+de


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


def get_repetitions(data, rate,
                    frame_length=0.05,
                    threshold_k=3,
                    hop_ratio=4,
                    window_size=0.5,
                    merge_distance_threshold=0.5,
                    min_final_length=2,
                    val_threshold_k=1.5,
                    max_near_distance=0.5):

    frame_sz = round(frame_length * rate)

    mtx, times = signal_windows_match_matrix_stft(
        data, frame_sz, rate, hop_ratio)

    frames = lambda sec: round(sec / frame_length * hop_ratio)
    seconds = lambda idx: times[idx]

    # adjust the magic number together with threshold_k
    threshold = 62.5e-6 * frame_sz * threshold_k

    for mt1, mt2, mlen in \
        filter_near(
            filter_self_overlapping(
                filter_by_length(
                    find_matches(
                        mtx, threshold,
                        frames(window_size),
                        frames(merge_distance_threshold),
                        threshold * val_threshold_k),
                    frames(min_final_length))),
            frames(max_near_distance)):
        yield seconds(mt1), seconds(mt2), seconds(mlen)
