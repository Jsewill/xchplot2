#!/usr/bin/env python3
"""Experimental multi-member PR #118 jobs; default xchplot2 output is unchanged."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile

def sync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save(path, text):
    """Publish an immutable, private job artifact before any plotting starts."""
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".partial.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.stat().st_size != len(text.encode()) or path.read_text(encoding="utf-8") != text:
                raise RuntimeError(f"different saved job at {path}")
        if os.name == "posix":
            os.chmod(path, 0o600)
        sync_directory(path.parent)
    finally:
        os.unlink(temporary)


def quoted(value):
    # std::quoted / current BatchManifest escaping, without shell interpolation.
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'


def run(command, **kwargs):
    return subprocess.run([str(arg) for arg in command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plotter", type=Path)
    parser.add_argument("reference", type=Path, help="built pos2_pr118_group")
    parser.add_argument("--out", required=True, type=Path, help="experimental .gplot destination")
    parser.add_argument("--farmer-pk", required=True)
    pool = parser.add_mutually_exclusive_group(required=True)
    pool.add_argument("--pool-pk")
    pool.add_argument("--pool-ph")
    parser.add_argument("--seed", help="32-byte seed in hex; random by default")
    parser.add_argument("--k", type=int, default=28)
    parser.add_argument("--strength", type=int, default=2)
    parser.add_argument("--meta-group", type=int, default=0)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--tier", default="plain", choices=["plain", "compact", "minimal", "tiny", "pinned", "auto"])
    parser.add_argument("--max-host-ram", default="18G", help="forwarded to plotter")
    parser.add_argument("--max-group-ram", type=int, default=512, metavar="MIB", help="modeled assembly/reader buffer limit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--testnet", action="store_true", help="unsupported in PR #118")
    args = parser.parse_args()
    if args.testnet:
        parser.error("PR #118 removes testnet mode; it cannot be used for grouped output")
    if not 18 <= args.k <= 28 or args.k % 2 or not 1 <= args.group_size <= 65535:
        parser.error("even k in 18..28 and group size in 1..65535 required")
    if not 0 <= args.meta_group <= 255 or not 2 <= args.strength <= args.k - 3:
        parser.error("invalid strength or meta group")
    if not 16 <= args.max_group_ram <= 1048576:
        parser.error("--max-group-ram must be 16..1048576 MiB")
    key = args.pool_pk if args.pool_pk is not None else args.pool_ph
    try:
        farmer = bytes.fromhex(args.farmer_pk)
        pool_key = bytes.fromhex(key)
        if len(farmer) != 48 or len(pool_key) != (48 if args.pool_pk is not None else 32):
            raise ValueError("invalid key length")
        if args.seed is not None and len(bytes.fromhex(args.seed)) != 32:
            raise ValueError("invalid seed length")
        if args.seed is not None:
            args.seed = bytes.fromhex(args.seed).hex()
    except ValueError as error:
        parser.error(str(error))
    plotter, reference, output = args.plotter.resolve(), args.reference.resolve(), args.out.resolve()
    if not plotter.is_file() or not reference.is_file() or output.suffix != ".gplot":
        parser.error("executable paths must exist and --out must end in .gplot")
    if any("\n" in str(path) or "\r" in str(path) for path in [output, plotter, reference]):
        parser.error("paths cannot contain line breaks")
    output.parent.mkdir(parents=True, exist_ok=True)
    job_path = Path(str(output) + ".job.json")
    raw_dir = Path(str(output) + ".raw")
    revision = run([reference, "identity"], capture_output=True, text=True).stdout.strip()
    if not revision or "source-sha256:" not in revision:
        raise RuntimeError("group reference returned no source identity")
    request = dict(revision=revision, k=args.k, strength=args.strength,
                   meta_group=args.meta_group, count=args.group_size,
                   farmer=farmer.hex(), pool=pool_key.hex(), output=str(output))
    if output.exists() and not args.resume:
        parser.error("group already exists; use --resume to validate it")
    if args.resume:
        if not job_path.is_file():
            parser.error("--resume requires the original saved group job")
        if job_path.stat().st_size > args.group_size * 256 + 4096:
            parser.error("saved group job exceeds the membership size bound")
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job["request"] != request or (args.seed is not None and job["seed"] != args.seed):
            parser.error("saved group job does not match the requested keys, parameters, or seed")
    else:
        job = dict(request=request, seed=args.seed if args.seed is not None else secrets.token_hex(32))
    # Re-derive and compare all identities on every resume; never trust edited IDs.
    prepared = run([reference, "prepare", args.k, args.strength, args.meta_group, args.group_size],
                   input=f"{job['seed']} {farmer.hex()} {pool_key.hex()}\n",
                   capture_output=True, text=True).stdout.splitlines()
    group_id, memo = prepared[0].split()
    ids = prepared[1:]
    if len(ids) != args.group_size or any(
            identity != hashlib.sha256(bytes.fromhex(group_id) + index.to_bytes(2, "big")
                                       + bytes([args.meta_group])).hexdigest()
            for index, identity in enumerate(ids)):
        raise RuntimeError("reference returned inconsistent member identities")
    derived = dict(group_id=group_id, memo=memo, members=ids)
    if args.resume and job["derived"] != derived:
        raise RuntimeError("saved group identity/memo/membership differs from key derivation")
    job["derived"] = derived
    save(job_path, json.dumps(job, sort_keys=True, indent=2) + "\n")
    raw_dir.mkdir(mode=0o700, exist_ok=True)
    if os.name == "posix":
        os.chmod(raw_dir, 0o700)
    sync_directory(output.parent)
    paths = [raw_dir / f"{identity}.plot2" for identity in ids]
    manifest = raw_dir / "batch.tsv"
    save(manifest, "# private experimental group members\n" + "".join(
        f"{args.k} {args.strength} {index} {args.meta_group} false {identity} {memo} "
        f"{quoted(raw_dir)} {quoted(path.name)}\n"
        for index, (identity, path) in enumerate(zip(ids, paths))))
    inputs = raw_dir / "group.inputs"
    save(inputs, f"{args.k} {args.strength} {args.meta_group} {group_id} {memo}\n"
         + "".join(quoted(path) + "\n" for path in paths))
    config = raw_dir / "empty.conf"
    save(config, "")
    run([reference, "check-raw", inputs, "-", args.max_group_ram])
    if output.exists():
        run([reference, "verify", inputs, output, args.max_group_ram])
    else:
        command = [plotter, "batch", manifest, "--config", config, "--devices", args.devices,
                   "--tier", args.tier, "--max-host-ram", args.max_host_ram,
                   "--quiet", "--no-progress"]
        if args.resume:
            command.append("--resume")
        run(command)
        fd, temporary = tempfile.mkstemp(prefix=output.name + ".partial.", dir=output.parent)
        os.close(fd)
        try:
            run([reference, "assemble", inputs, temporary, args.max_group_ram])
            # Windows requires a writable handle for fsync.
            with open(temporary, "r+b") as file:
                os.fsync(file.fileno())
            # Never replace a concurrent group; verify it if another writer won.
            try:
                os.link(temporary, output)
            except FileExistsError:
                run([reference, "verify", inputs, output, args.max_group_ram])
            sync_directory(output.parent)
        finally:
            os.unlink(temporary)
    print(output)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"experimental group stage failed (exit {error.returncode})") from None
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        raise SystemExit(f"experimental group job failed: {error}") from None
