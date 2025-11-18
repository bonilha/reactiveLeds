#!/usr/bin/env python3
"""
Data Collector for Reactive Audio → HDF5

- Listens to mirrored UDP packets (A1/A2) coming from pc-audio-linux.py
- Builds supervised examples using a sliding window:
    X: last 16 frames (T_in=16)
    Y: next 8 frames (T_out=8)
- Saves to chunked HDF5 with extendable datasets.

Usage:
  python data_collector.py --listen 0.0.0.0 --port 9000 --out data_b{BANDS}.h5

Notes:
- Packet formats follow the sender/receiver code:
  A1: [0]=0xA1, [1:9]=ts_pc_ns (u64), [9:9+nb]=bands u8, [..]=beat u8, [..]=transition u8
  A2: [0]=0xA2, [1:9]=ts_pc_ns (u64), [9:9+nb]=bands u8, [..]=beat u8, [..]=transition u8,
      [..]=dyn_floor u8, [..]=kick u8
"""

import argparse, socket, time, os
from collections import deque
from typing import Optional, Tuple

import numpy as np
try:
    import h5py  # pip install h5py
except Exception as e:
    h5py = None
    print("[WARN] h5py não está instalado. Instale com: pip install h5py")

PKT_AUDIO_V2 = 0xA2
PKT_AUDIO    = 0xA1

class SlidingBuilder:
    def __init__(self, bands_len: int, t_in: int = 16, t_out: int = 8):
        self.bands_len = int(bands_len)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.buf_b = deque()  # (bands_u8, beat, trans, dyn, kick)
        self.count = 0

    def push(self, bands_u8: np.ndarray, beat: int, trans: int, dyn: int, kick: int):
        if bands_u8.size != self.bands_len:
            # reset if bands len changed unexpectedly
            self.buf_b.clear()
            self.bands_len = int(bands_u8.size)
        self.buf_b.append((bands_u8.copy(), int(beat), int(trans), int(dyn), int(kick)))
        # keep a reasonable history: t_in + t_out + margin
        while len(self.buf_b) > (self.t_in + self.t_out + 8):
            self.buf_b.popleft()

    def has_pair(self) -> bool:
        return len(self.buf_b) >= (self.t_in + self.t_out)

    def make_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Build one (X, Y) pair using an anchor so that future frames exist.
        We use the last index as t; X = [t-15..t], Y = [t+1..t+8].
        """
        n = len(self.buf_b)
        need = self.t_in + self.t_out
        if n < need:
            return None
        # Convert deque to arrays once
        arr_bands = np.stack([b for (b, _, _, _, _) in self.buf_b], axis=0)  # [N, B]
        # anchor t is at index t_in-1
        t = self.t_in - 1
        x = arr_bands[t-self.t_in+1:t+1, :]        # [16, B]
        y = arr_bands[t+1:t+1+self.t_out, :]       # [8,  B]
        return x.astype(np.uint8), y.astype(np.uint8)

class H5Writer:
    def __init__(self, path: str, bands_len: int, t_in: int = 16, t_out: int = 8):
        if h5py is None:
            raise RuntimeError("h5py não instalado")
        self.path = path
        self.bands_len = int(bands_len)
        self.t_in = int(t_in); self.t_out=int(t_out)
        self._init_file()
        self._n = 0

    def _init_file(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or '.', exist_ok=True)
        self.f = h5py.File(self.path, 'a')
        # Datasets: X [N, T_in, B], Y [N, T_out, B]
        if 'X' not in self.f:
            maxshape = (None, self.t_in, self.bands_len)
            self.dX = self.f.create_dataset('X', shape=(0, self.t_in, self.bands_len),
                                            maxshape=maxshape, chunks=(64, self.t_in, self.bands_len), dtype='uint8')
        else:
            self.dX = self.f['X']
        if 'Y' not in self.f:
            maxshape = (None, self.t_out, self.bands_len)
            self.dY = self.f.create_dataset('Y', shape=(0, self.t_out, self.bands_len),
                                            maxshape=maxshape, chunks=(64, self.t_out, self.bands_len), dtype='uint8')
        else:
            self.dY = self.f['Y']
        # Meta
        self.f.attrs['bands_len'] = self.bands_len
        self.f.attrs['t_in'] = self.t_in
        self.f.attrs['t_out'] = self.t_out
        self.f.attrs['created'] = time.time()

    def append(self, X: np.ndarray, Y: np.ndarray):
        n0 = self.dX.shape[0]
        n1 = n0 + 1
        self.dX.resize((n1, self.t_in, self.bands_len))
        self.dY.resize((n1, self.t_out, self.bands_len))
        self.dX[n0, :, :] = X
        self.dY[n0, :, :] = Y
        self._n += 1
        if (self._n % 128) == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush(); self.f.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--listen', type=str, default='0.0.0.0')
    ap.add_argument('--port', type=int, default=9000)
    ap.add_argument('--out', type=str, default=None, help='HDF5 output path (use {bands} placeholder optional)')
    ap.add_argument('--t_in', type=int, default=16)
    ap.add_argument('--t_out', type=int, default=8)
    ap.add_argument('--min_pairs_flush', type=int, default=8)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1<<22)
    sock.bind((args.listen, args.port))
    print(f"[UDP] Listening on {args.listen}:{args.port} (A1/A2 mirror)")

    builder: Optional[SlidingBuilder] = None
    writer: Optional[H5Writer] = None

    received = 0
    made = 0
    last_flush = time.time()

    try:
        while True:
            data, _ = sock.recvfrom(8192)
            if not data:
                continue
            hdr = data[0]
            if hdr not in (PKT_AUDIO, PKT_AUDIO_V2):
                continue
            # Packet layout
            ts_pc_ns = int.from_bytes(data[1:9], 'little')
            nb = len(data) - (11 if hdr==PKT_AUDIO else 13)
            if nb <= 0:
                continue
            bands = np.frombuffer(memoryview(data)[9:9+nb], dtype=np.uint8)
            if hdr == PKT_AUDIO:
                beat = data[9+nb]
                trans = data[10+nb]
                dyn = 0; kick = 0
            else:
                beat = data[9+nb]
                trans = data[10+nb]
                dyn = data[11+nb]
                kick = data[12+nb]
            received += 1
            if builder is None:
                builder = SlidingBuilder(bands_len=bands.size, t_in=args.t_in, t_out=args.t_out)
                # decide output path
                out_path = args.out or f"data_b{bands.size}.h5"
                writer = H5Writer(out_path, bands_len=bands.size, t_in=args.t_in, t_out=args.t_out)
                print(f"[H5] Writing to {out_path} | B={bands.size} T_in={args.t_in} T_out={args.t_out}")
            builder.push(bands, beat, trans, dyn, kick)
            if builder.has_pair():
                pair = builder.make_pair()
                if pair is not None:
                    X, Y = pair
                    assert writer is not None
                    writer.append(X, Y)
                    made += 1
            # periodic status
            now = time.time()
            if (now - last_flush) >= 0.5:
                print(f"\rRX={received} pairs={made}", end='', flush=True)
                last_flush = now
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[END]")
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
