import progressbar as pb


def simple_progressbar(title):
    widgets = [f"{title}: ", pb.Percentage(), ' ',
               pb.Bar(marker='#', left='[', right=']'),
               ' ', pb.ETA()]
    return pb.ProgressBar(widgets=widgets,
                          maxval=1,
                          redirect_stdout=True)

def subprogress(f, offset, length):
    if f is None:
        return None
    return lambda x: f(offset + x * length)
