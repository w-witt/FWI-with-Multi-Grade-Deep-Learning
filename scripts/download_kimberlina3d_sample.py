#!/usr/bin/env python3
"""Download a small sample of the 3D Kimberlina dataset (OpenFWI "Kimberlina-V1")
directly from DOE/NETL EDX -- no EDX account required.

The 3D Kimberlina data linked from https://openfwi-lanl.github.io/docs/data.html
lives on https://edx.netl.doe.gov/group/kimberlina-geophysical-data. The web UI
asks for a login, but the underlying resource files are served publicly at
https://edx.netl.doe.gov/resource/<id>/download and support HTTP range requests.
This script uses range requests to extract single files out of the multi-GB zip
archives, so a paired (velocity model, shot gathers) sample costs ~200 MB of
download instead of ~10 GB.

Data layout (from the EDX "Python scripts" submission):
  - Velocity models: vp_year<Y>.zip -> year<Y>_cut<C>.bin
      float32, reshape to (400, 400, 350), 10 m cells
  - Seismic data:    csg_year<Y>_cut<C>.zip -> csg_year<Y>_cut<C>_<S>.H@
      float32, reshape to (40, 40, 5001) per shot -- 40x40 surface geophone
      grid at 100 m spacing, 5001 samples, dt = 0.001 s, 25 shots per cut
  - Years: 0,1,2,5,10,...,200 (33 snapshots of the CO2 leakage simulation)
  - Cuts:  1..63 spatial sub-volumes per year (year 25 has 58)

Examples:
  python scripts/download_kimberlina3d_sample.py                 # year 2, cut 1, 3 shots
  python scripts/download_kimberlina3d_sample.py --year 95 --cut 7 --shots 1 13 25
  python scripts/download_kimberlina3d_sample.py --list          # show available years/cuts

Output: .npy files under data/kimberlina3d_sample/ (override with --out).

License: the Kimberlina/OpenFWI datasets are CC BY-NC-SA 4.0. Cite:
  Deng et al., "OpenFWI: Large-scale multi-structural benchmark datasets for
  full waveform inversion", NeurIPS 2022 Datasets and Benchmarks.
  Alumbaugh et al., "The Kimberlina synthetic multiphysics dataset for CO2
  monitoring investigations", Geoscience Data Journal, 2024.
"""

import argparse
import io
import json
import re
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

EDX = "https://edx.netl.doe.gov"
GROUP_API = EDX + "/api/3/action/group_package_show?id=kimberlina-geophysical-data&limit=100"
# Cap the size of a single HTTP range request; BufferedReader loops as needed.
MAX_REQUEST = 32 * 1024 * 1024

VEL_SHAPE = (400, 400, 350)   # 10 m cells
SHOT_SHAPE = (40, 40, 5001)   # 40x40 geophones @ 100 m, 5001 samples, dt = 1 ms


def _ssl_context():
    ctx = ssl.create_default_context()
    try:  # python.org macOS builds often lack system CAs; prefer certifi if present
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    return ctx


CTX = _ssl_context()


def _open(url, **kw):
    return urllib.request.urlopen(url, timeout=600, context=CTX, **kw)


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP using range requests."""

    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with _open(req) as r:
            if r.headers.get("Accept-Ranges") != "bytes":
                raise IOError(f"server does not support range requests: {url}")
            self.length = int(r.headers["Content-Length"])
        self.pos = 0
        self.fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.length + offset}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = min(len(b), MAX_REQUEST)
        if self.pos >= self.length or n == 0:
            return 0
        end = min(self.pos + n - 1, self.length - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with _open(req) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        self.fetched += len(data)
        return len(data)


def open_remote_zip(resource_id):
    raw = HttpRangeFile(f"{EDX}/resource/{resource_id}/download")
    return raw, zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1024 * 1024))


def fetch_catalog():
    """Map the Kimberlina EDX group: velocity zips by year, seismic zips by (year, cut)."""
    print("Querying EDX catalog (no account needed) ...")
    with _open(GROUP_API) as r:
        d = json.load(r)
    if not d.get("success"):
        sys.exit("EDX API request failed; try again or see " + EDX + "/group/kimberlina-geophysical-data")

    vel, seis = {}, {}
    for pkg in d["result"]:
        title = pkg["title"]
        if title.endswith("3D velocity models"):
            for res in pkg["resources"]:
                m = re.fullmatch(r"vp_year(\d+)\.zip", res["name"])
                if m:
                    vel[int(m.group(1))] = res["id"]
        elif "3D seismic data" in title:
            for res in pkg["resources"]:
                m = re.fullmatch(r"csg_year(\d+)_cut(\d+)\.zip", res["name"])
                if m:
                    seis[(int(m.group(1)), int(m.group(2)))] = res["id"]
    if not vel or not seis:
        sys.exit("Could not find 3D velocity/seismic resources in the EDX catalog; "
                 "the group layout may have changed. Browse: " + EDX + "/group/kimberlina-geophysical-data")
    return vel, seis


def extract_member(zf, raw, member, dest, shape):
    info = zf.getinfo(member)
    print(f"  {member}: {info.file_size / 1e6:.0f} MB uncompressed "
          f"({info.compress_size / 1e6:.0f} MB to download) ...", flush=True)
    with zf.open(member) as src:
        data = np.frombuffer(src.read(), dtype=np.float32).reshape(shape)
    np.save(dest, data)
    print(f"    -> {dest}  shape={data.shape}  vmin={data.min():.3g} vmax={data.max():.3g}")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=2, help="simulation year (default: 2)")
    ap.add_argument("--cut", type=int, default=1, help="spatial cut 1-63 (default: 1)")
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 13, 25],
                    help="shot indices 1-25 to download (default: 1 13 25)")
    ap.add_argument("--all-shots", action="store_true", help="download all 25 shots of the cut")
    ap.add_argument("--out", type=Path, default=Path("data/kimberlina3d_sample"),
                    help="output directory (default: data/kimberlina3d_sample)")
    ap.add_argument("--list", action="store_true", help="list available years/cuts and exit")
    args = ap.parse_args()

    vel, seis = fetch_catalog()

    if args.list:
        years = sorted(vel)
        print("Velocity-model years:", " ".join(map(str, years)))
        for y in years:
            cuts = sorted(c for (yy, c) in seis if yy == y)
            if cuts:
                print(f"  year {y}: seismic cuts {cuts[0]}..{cuts[-1]} ({len(cuts)} cuts)")
        return

    if args.year not in vel:
        sys.exit(f"year {args.year} not found; available: {sorted(vel)}")
    if (args.year, args.cut) not in seis:
        cuts = sorted(c for (y, c) in seis if y == args.year)
        sys.exit(f"cut {args.cut} not found for year {args.year}; available cuts: {cuts}")

    args.out.mkdir(parents=True, exist_ok=True)
    shots = range(1, 26) if args.all_shots else args.shots

    print(f"\nVelocity model (year {args.year}, cut {args.cut})")
    raw, zf = open_remote_zip(vel[args.year])
    extract_member(zf, raw, f"year{args.year}_cut{args.cut}.bin",
                   args.out / f"vp_year{args.year}_cut{args.cut}.npy", VEL_SHAPE)

    print(f"\nSeismic shot gathers (year {args.year}, cut {args.cut}, shots {list(shots)})")
    raw, zf = open_remote_zip(seis[(args.year, args.cut)])
    prefix = zf.namelist()[0].split("/")[0] + "/" if "/" in zf.namelist()[0] else ""
    for s in shots:
        extract_member(zf, raw, f"{prefix}csg_year{args.year}_cut{args.cut}_{s}.H@",
                       args.out / f"csg_year{args.year}_cut{args.cut}_shot{s}.npy", SHOT_SHAPE)

    print("\nDone. Load with e.g.:")
    print(f"  vp  = np.load('{args.out}/vp_year{args.year}_cut{args.cut}.npy')   # (400, 400, 350), 10 m cells")
    print(f"  csg = np.load('{args.out}/csg_year{args.year}_cut{args.cut}_shot{shots[0] if not args.all_shots else 1}.npy')  # (40, 40, 5001), dt = 1 ms")
    print("\nData: Kimberlina 1.2 (NETL EDX), CC BY-NC-SA 4.0 -- cite OpenFWI (NeurIPS 2022)")
    print("and Alumbaugh et al. 2024 (Geoscience Data Journal) if you use it.")


if __name__ == "__main__":
    main()
