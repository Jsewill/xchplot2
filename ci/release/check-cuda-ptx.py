#!/usr/bin/env python3
"""Check the pinned AdaptiveCpp CUDA selector against LLVM and NVIDIA tools."""
import argparse
import pathlib
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acpp", type=pathlib.Path)
    parser.add_argument("compiler")
    parser.add_argument("llc")
    parser.add_argument("nvcc")
    parser.add_argument("ptxas")
    args = parser.parse_args()
    source = (args.acpp / "src/runtime/cuda/cuda_queue.cpp").read_text()
    selector = source[source.index("unsigned select_ptx_version("):
                      source.index("void host_synchronization_callback(")]
    supported = {int(value) for value in re.findall(
        r"^compute_(\d+)$", subprocess.check_output(
            [args.nvcc, "--list-gpu-arch"], text=True), re.MULTILINE)}
    assert supported, "NVCC reported no GPU architectures"
    # Include LLVM-unknown targets and a future target on both host platforms.
    devices = sorted(supported | {88, 103, 107, 110, 121, 130})
    with tempfile.TemporaryDirectory(prefix="xchplot2-ptx-") as temporary:
        work = pathlib.Path(temporary)
        probe = work / "selector.cpp"
        binary = work / "selector.exe"
        probe.write_text("#include <array>\n#include <vector>\n#include <iostream>\n" +
                         selector + "\nint main() { for(unsigned sm : {" +
                         ",".join(f"{sm}u" for sm in devices) +
                         "}) { unsigned target = 0; auto ptx = select_ptx_version(sm, target);"
                         'std::cout << sm << " " << target << " " << ptx << "\\n"; }}\n')
        subprocess.run([args.compiler, "-std=c++17", str(probe), "-o", str(binary)],
                       check=True)
        selected = subprocess.check_output([str(binary)], text=True).splitlines()
        ir = work / "probe.ll"
        ir.write_text('target triple = "nvptx64-nvidia-cuda"\n'
                      'define void @probe() { ret void }\n')
        for line in selected:
            device, target, version = map(int, line.split())
            assert 0 < target <= device, f"sm_{target} exceeds GPU sm_{device}"
            ptx = work / f"sm_{device}.ptx"
            result = subprocess.run(
                [args.llc, "-march=nvptx64", f"-mcpu=sm_{target}",
                 f"-mattr=+ptx{version}", str(ir), "-o", str(ptx)],
                check=True, capture_output=True, text=True)
            assert "not a recognized" not in result.stderr, result.stderr
            if device in supported:
                subprocess.run([args.ptxas, f"--gpu-name=sm_{device}", str(ptx),
                                "-o", str(work / "probe.cubin")], check=True)
        assert len(selected) == len(devices), "Missing target selection results"
    print(f"CUDA PTX selection and assembly passed for {len(supported)} GPU targets")


if __name__ == "__main__":
    main()
