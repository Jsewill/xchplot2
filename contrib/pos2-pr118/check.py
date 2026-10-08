#!/usr/bin/env python3
"""Compare current xchplot2 output with the pinned PoS2 1.0 reference."""

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
    parser.add_argument("--k", type=int, choices=(18, 22, 28), default=18)
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
            manifest.write_text(f"{args.k} {strength} {index} {meta_group} raw-v2 "
                                f"{group_id.hex()} 01020304 {root} {path.name}\n")
            command = [str(plotter), "batch", str(manifest), "--config", str(config),
                       "--devices", "gpu0" if args.gpu else "cpu", "--tier", "plain",
                       "--quiet", "--no-progress"]
            plotted = subprocess.run(command, capture_output=True, text=True)
            if plotted.returncode:
                raise RuntimeError(plotted.stdout + plotted.stderr)

            # Inspect only our freshly generated, bounded fixtures. This is
            # not a migration utility for arbitrary existing plots.
            assert raw_header.size + 4 <= path.stat().st_size <= 8 * (1 << args.k)
            with path.open("rb") as source:
                data = source.read(raw_header.size + 4)
            assert raw_header.unpack_from(data) == (
                b"pos2", 2, group_id, args.k, strength, index, meta_group, 4)
            assert data[raw_header.size:raw_header.size + 4] == bytes([1, 2, 3, 4])
            check = [str(reference), str(path), plot_id.hex()]
            if index == 0:
                grouped = root / "native.gplot"
                manifest.write_text(f"{args.k} {strength} 0 {meta_group} gplot-v2 "
                                    f"{group_id.hex()} 01020304 {root} {grouped.name}\n")
                subprocess.run(command, check=True)
                check.append(str(grouped))
            subprocess.run(check, check=True)
    print("PoS2 1.0 compatibility fixtures passed (" + ("GPU" if args.gpu else "CPU") + ").")


if __name__ == "__main__":
    main()
