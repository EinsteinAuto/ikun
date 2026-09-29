import collections
import os
import sys
import traceback

import torch

_ON = os.environ.get("IKUN_TRACE", "0") == "1"
_DEPTH = int(os.environ.get("IKUN_TRACE_DEPTH", "3"))

ITEM = collections.Counter()
CAST = collections.Counter()
COPY = collections.Counter()

_orig_item = torch.Tensor.item
_orig_to = torch.Tensor.to
_orig_contig = torch.Tensor.contiguous
_orig_float = torch.Tensor.float
_orig_half = torch.Tensor.half


def _site():
    out = []
    for fr in traceback.extract_stack()[:-2][::-1]:
        if "ikun_probe" in fr.filename:
            continue
        out.append("%s:%d" % (os.path.basename(fr.filename), fr.lineno))
        if len(out) >= _DEPTH:
            break
    return "<".join(out)


def _item(self):
    s = _site()
    ITEM[s] += 1
    print("@@@@@@@@item_sync={} n={}@@@@@@@@".format(s, ITEM[s]), flush=True)
    return _orig_item(self)


def _to(self, *a, **kw):
    out = _orig_to(self, *a, **kw)
    if out.dtype != self.dtype:
        s = "{}:{}->{}".format(_site(), self.dtype, out.dtype)
        CAST[s] += 1
        print("@@@@@@@@cast={} n={}@@@@@@@@".format(s, CAST[s]), flush=True)
    return out


def _mk_cast(orig, name):
    def f(self, *a, **kw):
        out = orig(self, *a, **kw)
        if out.dtype != self.dtype:
            s = "{}:{}->{}[.{}()]".format(_site(), self.dtype, out.dtype, name)
            CAST[s] += 1
            print("@@@@@@@@cast={} n={}@@@@@@@@".format(s, CAST[s]), flush=True)
        return out
    return f


def _contiguous(self, *a, **kw):
    out = _orig_contig(self, *a, **kw)
    if out.data_ptr() != self.data_ptr():
        s = "{}:{}".format(_site(), tuple(self.shape))
        COPY[s] += 1
        print("@@@@@@@@copy={} n={}@@@@@@@@".format(s, COPY[s]), flush=True)
    return out


def install():
    if not _ON:
        return
    torch.Tensor.item = _item
    torch.Tensor.to = _to
    torch.Tensor.contiguous = _contiguous
    torch.Tensor.float = _mk_cast(_orig_float, "float")
    torch.Tensor.half = _mk_cast(_orig_half, "half")
    print("========probe=installed depth={}============".format(_DEPTH), flush=True)


def report():
    if not _ON:
        return
    print("\n========probe_report============", flush=True)
    for label, c in (("item_sync", ITEM), ("cast", CAST), ("copy", COPY)):
        print("========{}_total={}============".format(label, sum(c.values())), flush=True)
        for site, n in c.most_common(10):
            print("========{}  {:>5}x  {}============".format(label, n, site), flush=True)


def uninstall():
    torch.Tensor.item = _orig_item
    torch.Tensor.to = _orig_to
    torch.Tensor.contiguous = _orig_contig
    torch.Tensor.float = _orig_float
    torch.Tensor.half = _orig_half
