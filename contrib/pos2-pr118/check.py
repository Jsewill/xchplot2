#!/usr/bin/env python3
"""Compare current xchplot2 output with the pinned PR #118 reference."""

import argparse
import hashlib
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plotter", type=Path, help="existing xchplot2 executable")
    parser.add_argument("reference", type=Path, help="built pos2_pr118_check executable")
    parser.add_argument("--gpu", action="store_true", help="use gpu0 instead of CPU plotting")
    args = parser.parse_args()
    plotter, reference = args.plotter.resolve(), args.reference.resolve()
    if not plotter.is_file() or not reference.is_file():
        parser.error("both executable paths must exist")

    group_id = bytes(range(32))
    raw_header = struct.Struct("<4sB32sBBHBB")
    with tempfile.TemporaryDirectory(prefix="xchplot2-pr118-") as temporary:
        root = Path(temporary)
        config = root / "empty.conf"
        config.write_text("")
        for index, meta_group, strength in [(0, 0, 2), (1, 0, 2), (0x1234, 7, 4), (65535, 255, 2)]:
            plot_id = hashlib.sha256(group_id + index.to_bytes(2, "big") + bytes([meta_group])).digest()
            path = root / f"index-{index}.plot2"
            manifest = root / "fixture.manifest"
            manifest.write_text(f"18 {strength} {index} {meta_group} false "
                                f"{plot_id.hex()} 01020304 {root} {path.name}\n")
            command = [str(plotter), "batch", str(manifest), "--config", str(config),
                       "--devices", "gpu0" if args.gpu else "cpu", "--tier", "plain",
                       "--quiet", "--no-progress"]
            plotted = subprocess.run(command, capture_output=True, text=True)
            if plotted.returncode:
                raise RuntimeError(plotted.stdout + plotted.stderr)

            # Inspect only our freshly generated, bounded k=18 fixtures. This is
            # not a migration utility for arbitrary existing plots.
            assert raw_header.size + 4 <= path.stat().st_size <= 8 * 1024 * 1024
            data = bytearray(path.read_bytes())
            assert raw_header.unpack_from(data) == (
                b"pos2", 1, plot_id, 18, strength, index, meta_group, 4)
            assert data[raw_header.size:raw_header.size + 4] == bytes([1, 2, 3, 4])
            # Raw chunk encoding is unchanged. In a private copy, adapt the v1
            # header from plot ID to group ID and v2, then test the real payload.
            data[4] = 2
            data[5:37] = group_id
            adapted = root / f"index-{index}.raw"
            adapted.write_bytes(data)
            subprocess.run([str(reference), str(adapted), plot_id.hex()], check=True)
    print("PR #118 compatibility fixtures passed (" + ("GPU" if args.gpu else "CPU") + ").")


if __name__ == "__main__":
    main()
