#!/usr/bin/env python3
"""Exercise installed group commands, saved jobs, read-only plans and recovery."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import struct
import subprocess
import sys
import tempfile
import time

# Public BLS12-381 generator; all keys and output in this check are disposable.
FARMER = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
POOL = "42" * 32
SEED = "ab" * 32
HEADER = struct.Struct("<4sB32sBBHBQB")
# Public key from the pinned keygen code for SEED's directory-farming family 0,
# FARMER and POOL. The RPC fixture still checks each generated group's ID.
FARMING_PLOT_KEY = "8c31b3cca1d31b09bb0172b7654cb2ad9beaef7ae15d59a5e7131aa9162dc966350306c2e8a7096902dac1905c821ddc"


def replot_rpc_fixture(args):
    """Typed local RPC fixture; no running farmer or network is contacted."""
    assert len(args) in (5, 6) and args[0] == "--root-path" and args[2] == "rpc", args
    root = Path(args[1])
    fixture = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    service, endpoint = args[3:5]
    assert service in ("harvester", "full_node", "farmer", "solver"), service
    assert service == "harvester" or not fixture.get("source_only"), "source reader used as a target service"
    if endpoint == "get_version":
        if fixture.get(service + "_unavailable"):
            raise SystemExit(1)
        result = {"version": fixture.get(service + "_version", fixture.get("version", "fixture-chia-v2"))}
    elif endpoint == "get_network_info":
        result = {"network_name": "simulator0", "genesis_challenge": ("b1" if service == "solver" and fixture.get("solver_wrong_network") else "a1") * 32}
    elif endpoint == "get_connections":
        assert service in ("harvester", "full_node", "farmer", "solver")
        peers = []
        if service == "harvester" and not fixture.get("farmer_disconnected"):
            peers = [(3, 0, 8447, 8447, "11")]
        elif service == "farmer":
            if not fixture.get("farmer_disconnected"):
                peers.append((2, 8447, 0, 61934, "33"))
            peers.append((1, 8447, 8445 if fixture.get("node_mismatch") else 8444, 8444, "44"))
            if not fixture.get("solver_disconnected"):
                peers.append((8, 8447, 8666, 8666, "55"))
        elif service == "full_node":
            peers = [(3, 8444, 8447, 61935, "22")]
        elif service == "solver" and not fixture.get("solver_disconnected"):
            peers = [(3, 8666, 8449 if fixture.get("solver_mismatch") else 8447, 61936, "22")]
        result = {"connections": [{"type": role, "local_port": local, "peer_server_port": remote,
                                  "peer_port": port, "peer_host": "127.0.0.1", "node_id": node * 32,
                                  "creation_time": 1.25, "last_message_time": 0}
                                 for role, local, remote, port, node in peers]}
    elif endpoint == "get_state":
        assert service == "solver"
        result = {"started": not fixture.get("solver_stopped")}
    elif endpoint == "get_pool_state":
        assert service == "farmer"
        pool = "99" * 32 if fixture.get("pool_mismatch") else POOL
        result = {"pool_state": [{"p2_singleton_puzzle_hash": pool,
                                 "pool_config": {"p2_singleton_puzzle_hash": pool,
                                                 "launcher_id": "11" * 32, "target_puzzle_hash": "22" * 32,
                                                 "owner_public_key": FARMER,
                                                 "pool_url": "https://unsupported.example" if fixture.get("pool_external") else ""}}]}
    elif endpoint == "get_constants":
        assert service == "full_node"
        result = {"constants": {"GENESIS_CHALLENGE": "a1" * 32, "HARD_FORK2_HEIGHT": 100,
                                "MIN_PLOT_STRENGTH": 2, "MAX_PLOT_STRENGTH": 17, "PLOT_SIZE_V2": 18}}
    elif endpoint == "get_blockchain_state":
        assert service == "full_node"
        result = {"blockchain_state": {"sync": {"synced": True, "sync_mode": False},
                  "peak": {"prev_transaction_block_height": 99 if fixture.get("inactive") else 101,
                           "timestamp": 0 if fixture.get("stale") else int(time.time())}}}
    elif endpoint in ("get_plots", "get_harvesters", "get_harvesters_summary"):
        assert service == ("harvester" if endpoint == "get_plots" else "farmer")
        plots = []
        for directory in map(Path, fixture["roots"]):
            for path in ([] if fixture.get("target_only") else sorted(directory.glob("*.plot"))):
                with path.open("rb") as stream:
                    assert stream.read(19) == b"Proof of Space Plot"
                    plot_id = stream.read(32).hex()
                    size = stream.read(1)[0]
                stat = path.stat()
                plots.append({"filename": str(path), "plot_id": plot_id, "size": size,
                              "file_size": stat.st_size, "time_modified": int(stat.st_mtime),
                              "pool_contract_puzzle_hash": POOL, "pool_public_key": None,
                              "plot_public_key": FARMER})
            if fixture.get("unloaded") or fixture.get("source_only"):
                continue
            for path in sorted(directory.glob("*.plot2")):
                group, _, strength, members, meta, memo = header(path)
                assert memo[:64] == POOL and memo[64:160] == FARMER
                expected = hashlib.sha256(bytes([strength]) + bytes.fromhex(FARMING_PLOT_KEY + POOL)).hexdigest()
                assert group == expected, "fixture's public key no longer matches the seeded family"
                stat = path.stat()
                plots.append({"filename": str(path), "plot_id": group, "size": 0x80 | strength,
                              "file_size": stat.st_size, "time_modified": int(stat.st_mtime),
                              "pool_contract_puzzle_hash": POOL, "pool_public_key": None,
                              "plot_public_key": FARMER if fixture.get("wrong_key") else FARMING_PLOT_KEY,
                              "group_size": members, "meta_group": meta})
        if endpoint == "get_plots":
            result = {"plots": plots}
        else:
            if fixture.get("farmer_unloaded"):
                plots = [plot for plot in plots if plot["size"] < 0x80]
            if fixture.get("farmer_wrong_key"):
                for plot in plots:
                    if plot["size"] >= 0x80:
                        plot["plot_public_key"] = FARMER
            if fixture.get("farmer_wrong_group"):
                for plot in plots:
                    if plot["size"] >= 0x80:
                        plot["group_size"] += 1
            result = {"harvesters": [{"connection": {"node_id": ("66" if fixture.get("farmer_mismatch") else "33") * 32,
                                                     "host": "127.0.0.1", "port": 61934},
                                       "syncing": {} if fixture.get("farmer_syncing") else None,
                                       "last_sync_time": 0 if fixture.get("farmer_zero_sync") else 1.25,
                                       "plots": len(plots) if endpoint == "get_harvesters_summary" else plots}]}
    else:
        raise AssertionError(endpoint)
    print(json.dumps({"success": True, **result}))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def header(path):
    with path.open("rb") as stream:
        magic, version, group, k, strength, count, meta, index, memo_size = HEADER.unpack(stream.read(HEADER.size))
        memo = stream.read(memo_size)
    assert magic == b"PoS2" and version == 2 and count, path
    assert HEADER.size + memo_size < index < path.stat().st_size, path
    # Windows private ACLs are checked by the native file and membership fixtures.
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600, path
    return group.hex(), k, strength, count, meta, memo.hex()


def farming_plan_rows(body):
    # std::quoted permits literal newlines inside metadata strings. Preserve
    # complete serializer rows rather than treating those as row boundaries.
    rows = []
    start = 0
    quoted = escaped = False
    for offset, byte in enumerate(body):
        if quoted:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                quoted = False
        elif byte == ord('"'):
            quoted = True
        elif byte == ord("\n"):
            rows.append(body[start:offset + 1])
            start = offset + 1
    assert not quoted and start == len(body), "incomplete farming fixture row"
    return rows


def rewrite_plan(record, *, producer=None, legacy=False, model=None):
    body = record[:-65]
    assert record[-65:] == hashlib.sha256(body).hexdigest().encode() + b"\n"
    rows = farming_plan_rows(body)
    assert rows[0] == b"ordinary-group-farming-v6\n"
    if producer is not None:
        rows[1] = json.dumps(producer).encode() + b"\n"
    if model is not None:
        rows[2] = json.dumps(model).encode() + b"\n"
    if legacy:
        metadata = rows[8].split()
        assert len(metadata) == 2 and metadata[1] in (b"0", b"1")
        if metadata[1] == b"1":
            # v5 metadata ends after its assumptions row. A calibrated v6
            # profile has no legacy representation and must not lose scope.
            assert rows[13] == b"0\n", "scoped calibration cannot become a legacy fixture"
            del rows[13]
        rows[0] = b"ordinary-group-farming-v5\n"
        del rows[3]  # v6 key-derivation and numeric compatibility row
    rewritten = b"".join(rows)
    return rewritten + hashlib.sha256(rewritten).hexdigest().encode() + b"\n"


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mode, p.stat().st_size, p.stat().st_mtime_ns,
                                     digest(p) if p.is_file() else None)
            for p in root.rglob("*")}


def check(binary, root, reference=None):
    counter = 0

    def command(*args):
        return [str(binary), str(args[0]), "--config", os.devnull, *map(str, args[1:])]

    def run(*args, code=0, point=None, input=None, cwd=None, timeout=180):
        nonlocal counter
        env = dict(os.environ)
        env.pop("XCHPLOT2_GROW_FAILPOINT", None)
        if point:
            env["XCHPLOT2_GROW_FAILPOINT"] = point
        result = subprocess.run(command(*args), capture_output=True, text=True, encoding="utf-8", env=env,
                                input=input, cwd=cwd, timeout=timeout)
        counter += 1
        (root / f"{counter:03d}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert SEED not in result.stdout + result.stderr, "private seed exposed in diagnostics"
        assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)
        return result

    def verify(path):
        result = run("verify", path, "--full", "--trials", 100)
        found = re.search(r"(\d+) full proofs validated", (result.stdout + result.stderr).lower())
        assert found and int(found[1]) > 0, result

    keys = ["-f", FARMER, "--pool-ph", POOL]
    construction = ["-d", "cpu", "-M", "512M", "-j", 2,
                    "--window-bytes", "64K", "--checkpoint-bytes", "64K",
                    "--capacity-bytes", "64M", "--headroom-bytes", 0]
    for subcommand in [("plan",), ("plot",), ("fill",), ("grow",), ("batch",),
                       ("bench",), ("test",), ("verify",), ("devices",), ("parity-check",),
                       ("compute",), ("compute", "broker"), ("compute", "worker"),
                       *(("compute", command) for command in ("init", "invite", "join", "configure", "status"))]:
        assert "Usage:" in run(*subcommand, "--help").stdout + (root / f"{counter:03d}.log").read_text(encoding="utf-8")

    devices = run("devices").stdout
    assert "Host CPU plotter" in devices and "backend=" not in devices
    assert "backend=" in run("devices", "-v").stdout
    quiet_devices = run("devices", "-q").stdout
    assert "Host CPU plotter" in quiet_devices and "Visible devices" not in quiet_devices
    run("devices", "-q", "-v", code=1)

    # A structural legacy header is sufficient for inventory, but is not a
    # complete proof fixture. Opaque entries stay visible and protected.
    inventory_root = root / "existing farm"
    (inventory_root / "nested").mkdir(parents=True)
    legacy = (b"Proof of Space Plot" + bytes.fromhex("12" * 32) + bytes([18]) +
              struct.pack(">H", 4) + b"v1.0" + struct.pack(">H", 112) +
              bytes.fromhex(POOL) + bytes.fromhex(FARMER) + bytes.fromhex("9a" * 32))
    first_table = len(legacy) + 80
    legacy += b"".join(struct.pack(">Q", first_table + 8 * i) for i in range(10)) + bytes(128)
    (inventory_root / "legacy with spaces.plot").write_bytes(legacy)
    (inventory_root / "nested" / "unknown.plot2").write_bytes(b"unknown container")
    inventory_before = snapshot(inventory_root)
    inventory_file = root / "inventory.plan"
    inventory_text = run("plan", "--inventory", inventory_root, "-M", "128M",
                         "--save-plan", inventory_file, "--json").stdout
    catalog = json.loads(inventory_text)
    assert catalog["kind"] == "farm-inventory" and catalog["complete"]
    assert len(catalog["roots"]) == 1 and len(catalog["plots"]) == 2
    entries = {entry["relative_path"]: entry for entry in catalog["plots"]}
    assert entries["legacy with spaces.plot"]["header_validated"]
    assert entries["legacy with spaces.plot"]["algorithm"] == "pos1"
    assert entries["nested/unknown.plot2"]["status"] == "protected"
    assert all(entry["workload_id"] for entry in entries.values())
    assert all(not resource["capacity_known"] for resource in catalog["farm_resources"]["resources"])
    assert all(workload["cost"] == "unknown" for workload in catalog["farm_resources"]["workloads"])
    assert "9a" * 32 not in inventory_text and json.loads(inventory_file.read_text(encoding="utf-8")) == catalog
    saved_inventory = inventory_file.read_bytes(), inventory_file.stat().st_mtime_ns
    run("plan", "--inventory", inventory_root, "--save-plan", inventory_file, code=2)
    assert (inventory_file.read_bytes(), inventory_file.stat().st_mtime_ns) == saved_inventory
    assert not run("plan", "--inventory", inventory_root, "--quiet").stdout
    run("plan", "--inventory", code=2)
    run("fill", "--inventory", inventory_root, code=2)
    run("plan", "--inventory", inventory_root, "--metadata-only", code=2)
    unprojected = root / "unprojected-migration.plan"
    run("plan", inventory_root, "--replot", inventory_file, "--save-plan", unprojected,
        "--compute", "off", code=2)
    assert not unprojected.exists() and snapshot(inventory_root) == inventory_before
    print("PASS public farm inventory, protected unknown entries and read-only migration rejection", flush=True)

    # Small synthetic plots exercise reporting, not farming performance.
    bench_dir = root / "benchmark"
    bench_dir.mkdir()
    bench = ["bench", "-k", 18, "-n", 1, "--warmup", 0, "-d", "cpu", "--cpu-workers", 1,
             "-o", bench_dir, "--target-size", "0.01", "--json"]
    measured = run(*bench)
    detailed = run(*bench, "-v", "--no-progress")
    quiet = run(*bench, "-q", "--progress")
    reports = [json.loads(result.stdout) for result in (measured, detailed, quiet)]
    assert all(report.keys() == reports[0].keys() for report in reports)
    assert all(report["passes"][0].keys() == reports[0]["passes"][0].keys() for report in reports)
    assert "progress:" not in measured.stderr and "per-plot spread:" not in measured.stderr
    assert "per-plot spread:" in detailed.stderr and "progress:" not in detailed.stderr
    assert "progress:" in quiet.stderr and "steady-state:" in quiet.stderr and "warmup:" not in quiet.stderr
    assert all("test-only" in result.stderr and "test-size rate" in result.stderr for result in (measured, detailed, quiet))
    assert not list(bench_dir.iterdir())
    print("PASS command help, inventory and benchmark verbosity, independent progress and JSON fields", flush=True)

    preview = root / "preview"
    preview_peer = root / "preview peer"
    preview.mkdir()
    preview_peer.mkdir()
    options = [preview, preview_peer, "-k", 18, "--group-members", 2, "--max-total-members", 3, *construction]
    before = snapshot(preview), snapshot(preview_peer)
    farming_size = run("plan", preview, "--group-members", 1, "--max-total-members", 1,
                       "--max-groups", 1, "--capacity-bytes", "2G", "--headroom-bytes", 0).stdout
    assert "k28" in farming_size and "test only" not in farming_size
    planned = json.loads(run("plan", *options, "--json").stdout)
    filled = json.loads(run("fill", *options, "--plan-only", "--json").stdout)
    assert planned == filled and planned["feasible"]
    assert len(planned["destinations"]) == 2 and all(d["files"] == 2 for d in planned["destinations"])
    assert (snapshot(preview), snapshot(preview_peer)) == before, "preview created or changed files"
    if os.name != "nt":
        native = root / os.fsdecode(b"native-byte-\xff")
        native.mkdir()
        native_options = [native, *options[2:]]
        native_plan = run("plan", *native_options, "--json").stdout
        native_fill = run("fill", *native_options, "--plan-only", "--json").stdout
        assert json.loads(native_plan) == json.loads(native_fill)
        assert "native-byte-\\u00ff" in native_plan and not list(native.iterdir())
    for subcommand in [("plan",), ("fill", "--plan-only")]:
        refused = json.loads(run(*subcommand, *options, "--farming-deadline-ms", 1, "--json", code=2).stdout)
        assert not refused["feasible"] and (snapshot(preview), snapshot(preview_peer)) == before
    print("PASS plan/fill preview equivalence and no writes on infeasibility", flush=True)

    # A portable profile contains harvester inputs, independent of destination
    # assignments and plotting keys. Offline observations remain unavailable.
    profile_file = root / "harvester profile.json"
    exported = json.loads(run("plan", "--farming-cores", 3, "--farming-memory-bytes", "256M",
                              "--save-profile", profile_file, "--json").stdout)
    assert json.loads(profile_file.read_text(encoding="utf-8")) == exported["farming_profile"]
    assert "destination_candidates" not in exported
    profile_before = profile_file.read_bytes(), profile_file.stat().st_mtime_ns
    run("plan", "--save-profile", profile_file, code=2)
    assert (profile_file.read_bytes(), profile_file.stat().st_mtime_ns) == profile_before
    run("plan", *options, "--save-profile", root / "invalid profile.json", code=2)
    assert not (root / "invalid profile.json").exists()
    imported = json.loads(run("plan", *options, "--harvester", profile_file, "--json").stdout)
    assert imported["harvester_cores"] == 3 and imported["memory_bytes"] == 256 * 1024 * 1024
    assert imported["farming_profile"] == exported["farming_profile"]
    assert imported["assessment_scope"] == "this-job" and imported["outside_job_workload"] == "unmodeled"
    overridden = json.loads(run("plan", *options, "--harvester", profile_file, "--farming-cores", 2, "--json").stdout)
    assert overridden["harvester_cores"] == overridden["farming_profile"]["harvester_cores"] == 2
    tuned_profile = root / "tuned harvester.json"
    quiet_export = run("plan", "--harvester", profile_file, "--farming-cores", 2,
                       "--save-profile", tuned_profile, "--quiet")
    assert not quiet_export.stdout and not quiet_export.stderr
    assert json.loads(tuned_profile.read_text(encoding="utf-8")) == overridden["farming_profile"]
    profile_config = root / "harvester.toml"
    profile_config.write_text('[plan]\nharvester = ' + json.dumps(str(profile_file), ensure_ascii=False) + '\n', encoding="utf-8")
    configured = json.loads(run("plan", *options, "--config", profile_config, "--json").stdout)
    assert configured["farming_profile"] == imported["farming_profile"]
    configured_override = json.loads(run("plan", *options, "--config", profile_config,
                                         "--harvester", tuned_profile, "--json").stdout)
    assert configured_override["farming_profile"] == overridden["farming_profile"]
    chia_root = root / "offline chia"
    (chia_root / "config").mkdir(parents=True)
    (chia_root / "config/config.yaml").write_text("self_hostname: localhost\nharvester:\n  num_threads: 7\n", encoding="utf-8")
    offline_profile = json.loads(run("plan", "--harvester", "local", "--chia-root", chia_root,
                                    "--chia-executable", root / "missing-chia", "--json").stdout)["farming_profile"]
    assert offline_profile["lookup_workers"] == 7 and not offline_profile["inventory_available"]
    assert offline_profile["host_name"] and offline_profile["captured_at"]
    assert "unavailable" in offline_profile["inventory_status"]
    assert (snapshot(preview), snapshot(preview_peer)) == before
    help_text = run("plan", "--help").stdout
    full_help = run("plan", "--help-all").stdout
    assert "--save-plan" in help_text and "--harvester" in help_text and len(help_text) < len(full_help)
    concise = run("plan", *options).stdout
    detailed = run("plan", *options, "-v").stdout
    assert len(concise) < len(detailed) and "--harvester" in concise
    assert "k18 (test only)" in concise
    assert run("plan", *options, "--progress").stdout == concise
    assert run("plan", *options, "-v", "--no-progress").stdout == detailed
    assert json.loads(run("plan", *options, "-v", "--json").stdout) == planned

    # Save exactly the reviewed assignments, then publish the private job and
    # recover without the preview or keys. A second execution is stale.
    reviewed_profile = dict(exported["farming_profile"])
    reviewed_profile.update(source="manual harvester profile", captured_at="2026-11-03T00:00:00Z",
                            chia_version="fixture version", host_name='harvester é "quoted"',
                            config_status="declared inputs", inventory_status="observed",
                            detected_logical_cpus=8, detected_available_memory_bytes=16 * 1024**3,
                            inventory_available=True, inventory_v1_files=4, inventory_v2_files=5,
                            inventory_v2_members=6, inventory_failed_files=1, inventory_no_key_files=2,
                            assumptions=["Declared CPU allowance", 'Unicode é, quote " and slash \\',
                                         "First line\nSecond line"])
    profile_file.write_text(json.dumps(reviewed_profile, ensure_ascii=False) + "\n", encoding="utf-8")
    review = root / "reviewed farm.plan"
    review_a, review_b = root / "review a", root / "review b"
    review_a.mkdir()
    review_b.mkdir()
    review_options = [review_a, review_b, *options[2:]]
    review_before = snapshot(review_a), snapshot(review_b)
    reviewed = json.loads(run("plan", *review_options, "--harvester", profile_file, "--save-plan", review, "--json").stdout)
    assert reviewed["farming_profile"] == reviewed_profile
    assert review.is_file() and SEED not in review.read_text(encoding="utf-8")
    assert (snapshot(review_a), snapshot(review_b)) == review_before
    saved_review = review.read_bytes()
    run("plan", *review_options, "--save-plan", review, code=2)
    assert review.read_bytes() == saved_review
    inspected = json.loads(run("plan", review, "--json").stdout)
    assert inspected["destinations"] == reviewed["destinations"]
    assert inspected["farming_profile"] == reviewed["farming_profile"]
    assert json.loads(run("plan", "--from-plan", review, "--json").stdout) == inspected
    assert json.loads(run("plan", "-F", review, "--json").stdout) == inspected
    if os.name != "nt":
        # Type detection must retain the ordinary saved-plan reader's regular,
        # no-follow input boundary before attempting even a short header read.
        linked_review = root / "linked farm.plan"
        linked_review.symlink_to(review)
        fifo_review = root / "fifo farm.plan"
        os.mkfifo(fifo_review)
        try:
            run("plan", "--from-plan", linked_review, "--json", code=2, timeout=5)
            run("plan", "--from-plan", fifo_review, "--json", code=2, timeout=5)
            assert review.read_bytes() == saved_review
        finally:
            linked_review.unlink()
            fifo_review.unlink()
    joint_single = json.loads(run("plan", "--assess-jobs", review, "--json").stdout)
    assert joint_single["joint_assessment"] and joint_single["feasible"]
    assert not joint_single["current_load_checked"] and not joint_single["execution_plan"]
    assert joint_single["assessment_scope"] == "saved-job-forecasts"
    assert joint_single["jobs"][0]["harvester_profile"]["host_name"] == reviewed_profile["host_name"]
    assert "Saved forecasts only" in run("plan", "--assess-jobs", review).stdout
    peer = root / "joint peer"
    peer.mkdir()
    peer_review = root / "joint peer.plan"
    run("plan", peer, *options[2:], "--harvester", profile_file, "--save-plan", peer_review, "--json")
    joint = json.loads(run("plan", "--assess-jobs", review, peer_review, "--json", code=2).stdout)
    assert len(joint["jobs"]) == 2 and not joint["feasible"] and joint["certified_max_eligible_groups"] == 2
    run("plan", "--assess-jobs", review, review, "--json", code=2)
    run("plan", "--assess-jobs", review, "--metadata-only", code=2)
    run("plan", "--assess-jobs", review, "--save-plan", root / "joint mutation.plan", code=2)
    run("plan", "--assess-jobs", review, "--harvester", "local", code=2)
    run("fill", "--assess-jobs", review, code=2)
    assert not (root / "joint mutation.plan").exists() and not list(peer.iterdir())
    offline_review = root / "review b offline"
    review_b.rename(offline_review)
    try:
        metadata = json.loads(run("plan", "--metadata-only", review, "--json").stdout)
        assert metadata["metadata_only"] and not metadata["live_readiness_checked"]
        assert metadata["checkpoint_compatibility"] == "not-inspected" and metadata["plan_compatible"]
        assert metadata["farming_profile"] == reviewed_profile
        assert {entry["directory"] for entry in metadata["destinations"]} == {str(review_a), str(review_b)}
        assert "current farming readiness were not checked" in run("plan", "--metadata-only", "-F", review).stdout
        offline_joint = json.loads(run("plan", "--assess-jobs", review, "--json").stdout)
        assert offline_joint["offline_destinations"] == 1
        assert offline_joint["jobs"][0]["members"] == joint_single["jobs"][0]["members"]
        for override in (["--save-plan", root / "metadata mutation.plan"], ["--harvester", "local"],
                         ["--farming-cores", 1], ["--extend"]):
            run("plan", "--metadata-only", review, *override, code=2)
        run("fill", "--metadata-only", review, code=2)
        run("plan", "--metadata-only", code=2)
        assert review.read_bytes() == saved_review and not (root / "metadata mutation.plan").exists()
    finally:
        offline_review.rename(review_b)
    run("plan", review, review_a, "--json", code=2)
    run("plan", review, "--save-profile", root / "invalid profile.json", code=2)
    assert not (root / "invalid profile.json").exists()
    run("fill", "--from-plan", review, "--resume", *keys, *construction, code=2)
    run("fill", "--from-plan", review, *keys, "--farming-cores", 1, *construction, code=2)
    assert (snapshot(review_a), snapshot(review_b)) == review_before
    run("fill", review, *keys, *construction, "--quiet", point="farm-plan-published", code=86)
    run("fill", review, *keys, *construction, code=2)
    recovered_review = run("fill", "-r", review_b, "--compute", "off", "-d", "cpu", "-M", "512M", "--quiet")
    reviewed_paths = [Path(line) for line in recovered_review.stdout.splitlines()]
    assert len(reviewed_paths) == 4 and all(sum(header(p)[3] for p in reviewed_paths if p.parent == d) == 3
                                            for d in (review_a, review_b))
    frozen_profile = json.loads(run("plan", review_a, "--json").stdout)["farming_profile"]
    assert frozen_profile == reviewed["farming_profile"]
    completed = snapshot(review_a), snapshot(review_b)
    run("fill", "-r", review_a, "--compute", "off", "-d", "cpu", "--quiet")
    assert (snapshot(review_a), snapshot(review_b)) == completed
    refreshed = root / "completed job forecast.plan"
    run("plan", "--manifest", review_a / "grow-farm.plan", "--save-plan", refreshed, "--json")
    refreshed_joint = json.loads(run("plan", "--assess-jobs", refreshed, "--json").stdout)
    assert refreshed_joint["jobs"][0]["members"] >= sum(header(path)[3] for path in reviewed_paths)
    assert refreshed_joint["jobs"][0]["source_job_digest"] == (review_a / "grow-farm.plan").read_bytes()[-65:-1].decode()
    assert (snapshot(review_a), snapshot(review_b)) == completed
    for path in reviewed_paths:
        verify(path)
    print("PASS portable profiles, offline detection, saved review execution and private recovery", flush=True)

    # Explicit reader calibration reuses a complete file and keeps the imported
    # harvester snapshot separate from this machine's bounded observations.
    calibration_plot = reviewed_paths[0]
    calibration_before = calibration_plot.stat(), digest(calibration_plot)
    calibrated_file = root / "calibrated harvester.json"
    calibrated = json.loads(run("plan", "--calibrate", "--harvester", profile_file,
                                "--calibration-samples", 2, "--calibration-timeout-ms", 10000,
                                "--save-profile", calibrated_file, "--json", "--", calibration_plot).stdout)
    assert json.loads(calibrated_file.read_text(encoding="utf-8")) == calibrated
    assert calibrated["reader_calibration"] and not calibrated["hardware_qualified"]
    calibrated_profile = calibrated["farming_profile"]
    scope = calibrated_profile["reader_calibration"]
    assert calibrated_profile["schema_version"] == 2 and (scope["k"], scope["strength"]) == header(calibration_plot)[1:3]
    assert scope["reader"] == calibrated_profile["reader"] and scope["method"] == "pos2-group-reference-reader-gsz-1"
    assert scope["host_name"] and scope["captured_at"]
    assert scope["host_name"] != reviewed_profile["host_name"] and scope["captured_at"] != reviewed_profile["captured_at"]
    for field in ("source", "captured_at", "host_name", "chia_version", "config_status", "inventory_status",
                  "detected_logical_cpus", "detected_available_memory_bytes", "inventory_available",
                  "inventory_v1_files", "inventory_v2_files", "inventory_v2_members", "inventory_failed_files", "inventory_no_key_files"):
        assert calibrated_profile[field] == reviewed_profile[field], field
    for field in ("chain_seconds_per_member", "index_seconds_per_entry"):
        assert calibrated_profile[field] >= reviewed_profile[field], field
    imported_calibration = root / "imported calibration.json"
    copied = json.loads(run("plan", "--harvester", calibrated_file, "-k", 18, "-s", 2,
                            "--save-profile", imported_calibration, "--json").stdout)["farming_profile"]
    assert copied == calibrated_profile == json.loads(imported_calibration.read_text(encoding="utf-8"))
    for mismatch in ([], ["-k", 18, "-s", 3], ["-k", 18, "--farming-reader", "cached"]):
        run("plan", "--harvester", calibrated_file, *mismatch, "--json", code=2)
    failed_calibration = root / "failed calibration.json"
    rejected = run("plan", "--calibrate", *keys, "--save-profile", failed_calibration,
                   "--json", "--", root / "missing calibration.gplot", code=2)
    assert "accept only their explicit inputs and controls" in rejected.stderr and not failed_calibration.exists()
    capped = run("plan", "--calibrate", "--calibration-worker", "--calibration-memory-bytes", 1024,
                 "--json", "--", root / "missing calibration.gplot", code=2)
    assert "startup" in capped.stderr and not capped.stdout
    run("plan", "--calibrate", "--calibration-samples", 1024, "--calibration-timeout-ms", 1,
        "--save-profile", failed_calibration, "--json", "--", calibration_plot, code=2)
    assert not failed_calibration.exists()
    after_calibration = calibration_plot.stat()
    before_info, before_digest = calibration_before
    assert (after_calibration.st_dev, after_calibration.st_ino, after_calibration.st_mode, after_calibration.st_size,
            after_calibration.st_mtime_ns, after_calibration.st_ctime_ns, digest(calibration_plot)) == (
            before_info.st_dev, before_info.st_ino, before_info.st_mode, before_info.st_size,
            before_info.st_mtime_ns, before_info.st_ctime_ns, before_digest)
    print("PASS bounded read-only reader calibration, scoped profile import and snapshot provenance", flush=True)

    # Compression admits one more member than the initial forecast. The next
    # candidate exceeds this I/O window; sealing must make resume a no-op.
    capacity_dir = root / "capacity fill"
    capacity_dir.mkdir()
    capacity_options = [capacity_dir, "-k", 18, *construction, "--compute", "off",
                        "--construction", "incremental", "--capacity-bytes", 5110000,
                        "--window-bytes", 1536, "--batch-members", 2]
    capacity_plan = json.loads(run("plan", *capacity_options, "--json").stdout)
    assert capacity_plan["destinations"][0]["members"] == 7
    capacity_fill = run("fill", *capacity_options, *keys, "--seed", "aa" * 32, "--json")
    capacity_metrics = json.loads(capacity_fill.stdout.splitlines()[-1])
    capacity_path, = capacity_dir.glob("*.gplot")
    assert header(capacity_path)[3] == capacity_metrics["members"] == 8
    assert capacity_metrics["stop_reason"] == "capacity"
    assert capacity_metrics["peak_file_bytes"] <= capacity_metrics["capacity_bytes"]
    completed = snapshot(capacity_dir)
    resumed = run("fill", "-r", *capacity_options, "--json")
    assert json.loads(resumed.stdout.splitlines()[-1])["generated_members"] == 0
    assert snapshot(capacity_dir) == completed, "completed capacity fill retried or changed files"
    verify(capacity_path)
    print("PASS compression top-up, window-limited sealing and unchanged resume", flush=True)

    output = root / "grouped files"
    plot = ["plot", "-k", 18, "-n", 2, *keys, "-o", output,
            "--group-members", 3, "--batch-members", 2, "--construction", "staged",
            *construction, "--quiet"]
    made = run(*plot)
    paths = [Path(line) for line in made.stdout.splitlines()]
    assert len(paths) == 2 and all(header(p)[3] == 3 for p in paths), made
    assert not made.stderr, made.stderr
    assert header(paths[0])[0] != header(paths[1])[0]
    job, = output.glob("xchplot2-job-*.tsv")
    saved = job.read_bytes()
    rows = [shlex.split(line) for line in saved.decode().splitlines() if line and not line.startswith("#")]
    assert len(rows) == 2 and all(row[4] == "gplot-v2-grouped" and row[9] == "3" for row in rows)
    if os.name != "nt":
        assert job.stat().st_mode & 0o777 == 0o600
    before = {path: digest(path) for path in paths}
    repeated = run(*plot, "--resume")
    assert set(repeated.stdout.splitlines()) == set(map(str, paths))
    assert job.read_bytes() == saved and all(digest(p) == value for p, value in before.items())
    # batch uses the saved membership even when no group-specific flags are repeated.
    run("batch", job, "--resume", *construction, "--quiet")
    assert all(digest(p) == value for p, value in before.items())
    run(*plot, "--resume", "--group-members", 4, code=2)
    run(*plot, "--group-members", 0, code=1)
    for path in paths:
        verify(path)
    fresh = run(*plot)
    assert not set(fresh.stdout.splitlines()) & set(map(str, paths)), "new jobs reused plot identities"
    print("PASS file counts, membership, private manifests, fresh identities, resume and full proofs", flush=True)

    # A one-member group must match the existing ordinary writer exactly.
    one = root / "one"
    single = root / "single"
    grouped_result = run("plot", "-k", 18, *keys, "-S", SEED, "-o", one,
                         "--group-members", 1, *construction, "--quiet", "--progress")
    grouped_one = Path(grouped_result.stdout.strip())
    assert "members assembled" in grouped_result.stderr
    assert "construction=" not in grouped_result.stderr and "committed " not in grouped_result.stderr
    ordinary_result = run("plot", "-k", 18, *keys, "-S", SEED, "-o", single,
                          "--devices", "cpu", "--cpu-workers", 1, "--quiet")
    ordinary_one = Path(ordinary_result.stdout.strip())
    assert not ordinary_result.stderr
    assert digest(grouped_one) == digest(ordinary_one)
    if reference:
        group, k, strength, count, meta, memo = header(paths[0])
        fragments = root / "reference fragments"
        fragments.mkdir()
        raw = []
        for member in range(count):
            name = f"{member}.plot2"
            run("test", k, group, strength, member, meta, "--raw", "-m", memo, "-o", fragments, "-N", name)
            raw.append(fragments / name)
        inputs = fragments / "group.inputs"
        inputs.write_text(f"{k} {strength} {meta} {group} {memo}\n" +
                          "".join(json.dumps(str(p), ensure_ascii=False) + "\n" for p in raw), encoding="utf-8")
        canonical = fragments / "canonical.gplot"
        comparison = subprocess.run([str(reference), "assemble-only", str(inputs), str(canonical), "512"],
                                    capture_output=True, text=True, encoding="utf-8", timeout=180)
        assert comparison.returncode == 0, (comparison.stdout, comparison.stderr)
        assert digest(canonical) == digest(paths[0])
    print("PASS ordinary byte parity", flush=True)

    # The native storage adapter must recover every construction engine to the
    # same ordinary bytes. Each failpoint exits a real child process abruptly.
    expected = None
    scratch = root / "engine scratch"
    scratch.mkdir()
    for mode, point in (("incremental", "initialized"), ("staged", "stage-run-committed"),
                        ("merged", "merge-index-flushed"), ("external", "external-header-flushed")):
        engine = ["plot", "-k", 18, *keys, "-S", SEED, "-o", root / (mode + " recovered"),
                  "--group-members", 3, "--batch-members", 2, "--construction", mode,
                  *construction, "-t", scratch, "--temp-capacity-bytes", "256M", "--quiet"]
        run(*engine, point=point, code=86)
        recovered = Path(run(*engine, "--resume").stdout.strip())
        assert header(recovered)[3] == 3
        if expected is None:
            expected = digest(recovered)
        assert digest(recovered) == expected, mode
        verify(recovered)
    print("PASS all construction engines recover to identical ordinary bytes and full proofs", flush=True)

    concise_grow = run("grow", grouped_one, "--max-members", 3, *construction)
    assert "[grow] 3 members" in concise_grow.stderr
    assert "members assembled" not in concise_grow.stderr and "Buffers:" not in concise_grow.stderr
    assert header(grouped_one)[3] == 3
    verify(grouped_one)
    grown = digest(grouped_one)
    detailed_grow = run("grow", grouped_one, "--max-members", 3, *construction, "-v", "--no-progress")
    assert "Backend:" in detailed_grow.stderr and "Buffers:" in detailed_grow.stderr
    assert "members assembled" not in detailed_grow.stderr
    assert digest(grouped_one) == grown
    run("grow", root / "missing.gplot", "--max-members", 2, *construction, code=2)
    assert not (root / "missing.gplot").exists()

    # Reject a corrupt saved identity before changing any peer output.
    corrupted = root / "mismatch.tsv"
    rows[0][5] = "cc" * 32
    corrupted.write_text("\n".join(" ".join(json.dumps(word, ensure_ascii=False) if index in (6, 7, 8) else word
                                            for index, word in enumerate(row)) for row in rows) + "\n", encoding="utf-8")
    run("batch", corrupted, "--resume", *construction, code=2)
    assert all(digest(p) == value for p, value in before.items())
    print("PASS existing-file growth and identity audit", flush=True)

    interrupted = root / "interrupted"
    recover = ["plot", "-k", 18, "-n", 2, *keys, "-o", interrupted,
               "--group-members", 3, "--batch-members", 1, "--construction", "staged", *construction, "--quiet"]
    run(*recover, point="stage-run-committed", code=86)
    recovery_job, = interrupted.glob("xchplot2-job-*.tsv")
    recovery_manifest = recovery_job.read_bytes()
    completed = run(*recover, "--resume")
    assert len(completed.stdout.splitlines()) == 2 and recovery_job.read_bytes() == recovery_manifest
    for name in completed.stdout.splitlines():
        assert header(Path(name))[3] == 3
        verify(Path(name))

    cancelled = root / "cancelled"
    cancel_args = ["plot", "-k", 22, "-n", 6, *keys, "-o", cancelled,
                   "--group-members", 2, "--batch-members", 1, "--construction", "staged",
                   *construction, "--capacity-bytes", "256M", "--verbose"]
    cancel_log = root / "cancel.log"
    with cancel_log.open("w+") as log:
        process = subprocess.Popen(command(*cancel_args), stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 30
            while ": generating cpu member" not in cancel_log.read_text(encoding="utf-8"):
                assert process.poll() is None, cancel_log.read_text(encoding="utf-8")
                assert time.monotonic() < deadline, cancel_log.read_text(encoding="utf-8")
                time.sleep(0.01)
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            output = process.communicate(timeout=180)[0]
            assert process.returncode == 4, (process.returncode, output, cancel_log.read_text(encoding="utf-8"))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    cancel_job, = cancelled.glob("xchplot2-job-*.tsv")
    cancel_manifest = cancel_job.read_bytes()
    run(*cancel_args, "--resume", "--no-verbose", "--quiet")
    assert cancel_job.read_bytes() == cancel_manifest
    assert len(list(cancelled.glob("*.gplot"))) == 6
    verify(next(cancelled.glob("*.gplot")))
    print("PASS process cancellation leaves a resumable grouped job", flush=True)

    fill = ["fill", *options, *keys, "--seed", SEED, "--quiet"]
    produced = run(*fill)
    farm_outputs = [Path(line) for line in produced.stdout.splitlines()]
    assert sorted(header(p)[3] for p in farm_outputs) == [1, 1, 2, 2]
    assert {p.parent for p in farm_outputs} == {preview, preview_peer}
    assert (preview / "grow-farm.plan").is_file()
    farm_hashes = {p: digest(p) for p in farm_outputs}
    assert set(run(*fill, "-r").stdout.splitlines()) == set(map(str, farm_outputs))
    assert all(digest(p) == value for p, value in farm_hashes.items())
    for path in farm_outputs:
        verify(path)
    print("PASS interrupted construction, directory fill, saved-plan resume and full proofs", flush=True)

    # An interrupted initial publication must retain its random identity, and
    # either member must locate the complete job without repeated keys/options.
    farm = root / "saved farm"
    farm.mkdir()
    members = [farm / name for name in ("disk a", "disk b", "disk c")]
    for directory in members:
        directory.mkdir()
    a, b, c = members
    initial = ["fill", *keys, "-k", 18, "--group-members", 2, "--max-total-members", 3,
               "--max-groups", 2, "--farming-max-eligible-groups", 4, *construction,
               "--construction", "incremental", "--quiet", "--", a, b]
    run(*initial, point="farm-plan-published", code=86)
    resume = ["fill", "-r", "-d", "cpu", "-M", "512M", "-j", 2, "--compute", "off", "--quiet"]
    recovered = run(*resume, b)
    farm_paths = [Path(line) for line in recovered.stdout.splitlines()]
    assert len(farm_paths) == 4 and {p.parent for p in farm_paths} == {a, b}
    original_plan = (a / "grow-farm.plan").read_bytes()
    assert original_plan == (b / "grow-farm.plan").read_bytes()
    # A compatible creating build is provenance. The normal executable must
    # still audit saved keys/identities and completed ordinary output files.
    foreign_producer = "e" * 64
    original_plan = rewrite_plan(original_plan, producer=foreign_producer)
    for directory in (a, b):
        (directory / "grow-farm.plan").write_bytes(original_plan)
    metadata = json.loads(run("plan", "--metadata-only", "--manifest", a / "grow-farm.plan", "--json").stdout)
    assert metadata["producer_build"] == foreign_producer and metadata["plan_compatible"]
    legacy = root / "legacy foreign farm.plan"
    legacy.write_bytes(rewrite_plan(original_plan, legacy=True))
    legacy_before = snapshot(a), snapshot(b), legacy.read_bytes()
    legacy_metadata = json.loads(run("plan", "--metadata-only", "--manifest", legacy, "--json").stdout)
    assert legacy_metadata["schema_version"] == 5 and not legacy_metadata["plan_compatible"]
    run("plan", "--manifest", legacy, "--json", code=2)
    assert (snapshot(a), snapshot(b), legacy.read_bytes()) == legacy_before
    def checksummed(rows):
        body = b"".join(rows)
        return body + hashlib.sha256(body).hexdigest().encode() + b"\n"

    def quoted(value):
        return b'"' + value.encode().replace(b"\\", b"\\\\").replace(b'"', b'\\"') + b'"'

    rows = farming_plan_rows(original_plan[:-65])
    assert rows[8].split()[1] == b"1" and rows[13] == b"0\n"
    no_metadata = list(rows)
    no_metadata[8] = no_metadata[8].split()[0] + b" 0\n"
    del no_metadata[9:14]
    multiline = list(rows)
    extra_assumption = 'line one\n0\nline two "quoted" and \\ slash'
    count, _, assumptions = multiline[12].partition(b" ")
    multiline[12] = (str(int(count) + 1).encode() + b" " + assumptions.removesuffix(b"\n") + b" " +
                     quoted(extra_assumption) + b"\n")
    for name, variant in (("without metadata", no_metadata), ("multiline metadata", multiline)):
        variant_path = root / ("legacy " + name + ".plan")
        variant_path.write_bytes(rewrite_plan(checksummed(variant), legacy=True))
        inspected_variant = json.loads(run("plan", "--metadata-only", "--manifest", variant_path, "--json").stdout)
        assert inspected_variant["schema_version"] == 5 and not inspected_variant["plan_compatible"]
        if name == "without metadata":
            assert inspected_variant["farming_profile"] is None
        else:
            assert inspected_variant["farming_profile"]["assumptions"][-1] == extra_assumption
    scoped = list(rows)
    scoped[13] = (b"1 18 2 0 " + quoted(scope["method"]) + b" " + str(scope["margin"]).encode() + b" " +
                  quoted(scope["host_name"]) + b" " + quoted(scope["captured_at"]) + b"\n")
    try:
        rewrite_plan(checksummed(scoped), legacy=True)
    except AssertionError as error:
        assert "scoped calibration" in str(error)
    else:
        raise AssertionError("legacy fixture conversion discarded reader calibration scope")
    unsupported = root / "unsupported model farm.plan"
    unsupported.write_bytes(rewrite_plan(original_plan, model="unknown-farming-model"))
    run("plan", "--metadata-only", "--manifest", unsupported, "--json", code=2)
    for directory in (a, b):
        assert not (directory / "grow-farm.plan.next").exists()
        if os.name != "nt":
            assert (directory / "grow-farm.plan").stat().st_mode & 0o777 == 0o600
    before = snapshot(a), snapshot(b)
    assert set(run(*resume, b, a).stdout.splitlines()) == set(map(str, farm_paths))
    assert set(run(*resume, "--manifest", b / "grow-farm.plan").stdout.splitlines()) == set(map(str, farm_paths))
    assert set(run(*resume, cwd=b).stdout.splitlines()) == set(map(str, farm_paths))
    assert (snapshot(a), snapshot(b)) == before, "keyless recovery rewrote the saved job or completed plots"
    for path in farm_paths:
        verify(path)
    run(*resume, b, "-S", "cd" * 32, code=2)
    assert (snapshot(a), snapshot(b)) == before
    for changed in (["--max-total-members", 4], ["--max-groups", 3], ["--capacity-bytes", "96M"]):
        run(*resume, b, *changed, code=2)
        assert (snapshot(a), snapshot(b)) == before, "changed extent mutated the saved job"

    candidates = json.loads(run("plan", "--json").stdout)
    assert candidates["destination_candidates"] and isinstance(candidates["destinations"], list)
    paths_file = farm / "destinations.txt"
    paths_file.write_text(f"{b}\n{a}\n", encoding="utf-8")
    selected = json.loads(run("plan", "--paths-from", paths_file, "--json").stdout)
    streamed = json.loads(run("plan", "--paths0-from", "-", "--json", input=f"{a}\0{b}\0").stdout)
    assert selected == streamed and len(selected["destinations"]) == 2
    run("plan", "--paths-from", "-", input="", code=2)
    print("PASS private replicated jobs, path lists and keyless recovery from any member", flush=True)

    # Publication precedes opening the compute connection. Preserve that saved
    # owner namespace when recovery switches to a local producer.
    owned, control = farm / "compute owner", farm / "owner control"
    owned.mkdir()
    control.mkdir()
    owner = "farm-owner"
    owned_options = [*keys, "-k", 18, "--group-members", 1, "--max-total-members", 1,
                     "--max-groups", 1, *construction, "--quiet"]
    run("fill", *owned_options, "-S", SEED, "--compute-broker", "127.0.0.1:1",
        "--compute-owner-id", owner, "--compute-ca", farm / "unused-ca",
        "--compute-cert", farm / "unused-cert", "--compute-key", farm / "unused-key",
        owned, point="farm-plan-published", code=86)
    owned_path = Path(run(*resume, owned).stdout.strip())
    scoped_seed = hashlib.sha256(b"xchplot2-pool-storage-owner-v1" + bytes.fromhex(SEED) + owner.encode()).hexdigest()
    control_path = Path(run("fill", *owned_options, "--compute", "off", "-S", scoped_seed, control).stdout.strip())
    assert digest(owned_path) == digest(control_path), "changing compute provider changed saved plot identities"
    verify(owned_path)
    print("PASS keyless local recovery preserves the saved compute owner namespace", flush=True)

    offline = farm / "disk b offline"
    b.rename(offline)
    before_a = snapshot(a)
    run(*resume, a, code=2)
    partial = run(*resume, "--allow-missing", a, code=4)
    assert set(partial.stdout.splitlines()) == {str(p) for p in farm_paths if p.parent == a}
    assert "unavailable" in partial.stderr and snapshot(a) == before_a
    unavailable = json.loads(run("plan", "--json", a, code=2).stdout)
    assert [d["available"] for d in unavailable["destinations"]] == [True, False]
    # An empty directory at an absent mountpoint is not the saved volume.
    b.mkdir()
    run(*resume, a, code=2)
    assert not list(b.iterdir())
    b.rmdir()
    offline.rename(b)
    assert set(run(*resume, a).stdout.splitlines()) == set(map(str, farm_paths))
    print("PASS unavailable members remain pending and replaced mountpoints receive no writes", flush=True)

    # Expansion on shared storage also accounts for the existing family's
    # workload. Completed fixed extents release their unused capacity quota.
    original_hashes = {p: digest(p) for p in farm_paths}
    before = tuple(snapshot(directory) for directory in members)
    expansion_review = farm / "expansion.plan"
    expansion = json.loads(run("plan", c, b, a, "--save-plan", expansion_review, "--json").stdout)
    assert expansion["feasible"] and len(expansion["destinations"]) == 3
    assert tuple(snapshot(directory) for directory in members) == before
    run(*resume, a, c, code=2)
    assert tuple(snapshot(directory) for directory in members) == before
    assert SEED not in expansion_review.read_text(encoding="utf-8")
    run("fill", "-F", expansion_review, "-d", "cpu", "-M", "512M", "-j", 2, "--quiet",
        point="farm-plan-published:2", code=86)
    assert all(digest(p) == value for p, value in original_hashes.items())
    extended = run(*resume, c)
    all_paths = [Path(line) for line in extended.stdout.splitlines()]
    assert len(all_paths) == 6 and {p.parent for p in all_paths} == set(members)
    assert all(digest(p) == value for p, value in original_hashes.items())
    revised = (c / "grow-farm.plan").read_bytes()
    assert revised != original_plan and all((d / "grow-farm.plan").read_bytes() == revised for d in members)
    identities = {(header(p)[0], header(p)[4], index) for p in all_paths for index in range(header(p)[3])}
    assert len(identities) == sum(header(p)[3] for p in all_paths), "expansion reused an existing plot identity"
    for path in all_paths:
        if path.parent == c:
            verify(path)
    # Mixed direct-parent publication was recovered above. Older ancestors no
    # longer describe the sealed extents and cannot replace the current job.
    (b / "grow-farm.plan").write_bytes(original_plan)
    run(*resume, b, code=2)
    (b / "grow-farm.plan").write_bytes(revised)
    run("fill", "--from-plan", expansion_review, "-d", "cpu", "--quiet", code=2)
    corrupt = bytearray(revised)
    corrupt[-2] = ord("0") if corrupt[-2] != ord("0") else ord("1")
    (b / "grow-farm.plan").write_bytes(corrupt)
    before_a = snapshot(a)
    run(*resume, a, code=2)
    assert snapshot(a) == before_a
    (b / "grow-farm.plan").write_bytes(revised)
    print("PASS expansion preserves old plots, accounts for the whole farm and recovers partial publication", flush=True)

    relocated = farm / "disk a moved"
    a.rename(relocated)
    run(*resume, b, code=2)
    before = tuple(snapshot(directory) for directory in (relocated, b, c))
    relocation_review = farm / "relocation.plan"
    moving = json.loads(run("plan", "--relocate", a, relocated, "--save-plan", relocation_review, "--json").stdout)
    assert moving["feasible"] and moving["destinations"][0]["directory"] == str(relocated)
    assert tuple(snapshot(directory) for directory in (relocated, b, c)) == before
    moved = run("fill", "--from-plan", relocation_review, "--compute", "off", "-d", "cpu", "-M", "512M", "--quiet")
    moved_paths = [Path(line) for line in moved.stdout.splitlines()]
    assert len(moved_paths) == 6 and all(p.exists() for p in moved_paths)
    for path, value in original_hashes.items():
        assert digest(relocated / path.name if path.parent == a else path) == value
    assert (relocated / "grow-farm.plan").read_bytes() == (c / "grow-farm.plan").read_bytes()
    run(*resume, c)
    other_job = farm / "different job"
    other_job.mkdir()
    run("fill", *keys, "-k", 18, "--max-total-members", 1, "--max-groups", 1, *construction, "--quiet", other_job)
    before = snapshot(relocated), snapshot(other_job)
    run("plan", relocated, other_job, code=2)
    assert (snapshot(relocated), snapshot(other_job)) == before
    print("PASS explicit relocation and conflicting-job rejection", flush=True)

    # Recovery also completes an interrupted preparation before any replica
    # is published. An offline member cannot commit an incomplete preparation.
    for point in ("farm-plan-before-publication", "farm-plan-all-staged", "initialized"):
        interrupted = farm / point
        interrupted.mkdir()
        left, right = interrupted / "left", interrupted / "right"
        left.mkdir()
        right.mkdir()
        run("fill", *keys, "-k", 18, "--group-members", 2, "--max-total-members", 1,
            "--max-groups", 1, *construction, "--construction", "incremental", "--quiet", left, right,
            point=point, code=86)
        if point == "farm-plan-before-publication":
            # The controller failpoint follows joined staging. Remove one
            # copy to retain recovery coverage for a partially staged job.
            assert (left / "grow-farm.plan.next").read_bytes() == (right / "grow-farm.plan.next").read_bytes()
            (right / "grow-farm.plan.next").unlink()
        pending = interrupted / "offline"
        right.rename(pending)
        before = snapshot(left)
        if point != "initialized":
            run(*resume, "--allow-missing", left, code=2)
            assert snapshot(left) == before
        else:
            available = run(*resume, "--allow-missing", left, code=4)
            assert not available.stdout.strip() and "waits for all saved farming destinations" in available.stderr
        pending.rename(right)
        if point == "farm-plan-before-publication":
            run(*resume, right, code=2)  # This member has no recovery copy.
        finished = run(*resume, left if point == "farm-plan-before-publication" else right)
        paths = [Path(line) for line in finished.stdout.splitlines()]
        assert len(paths) == 2 and all(header(path)[3] == 1 for path in paths)
        assert (left / "grow-farm.plan").read_bytes() == (right / "grow-farm.plan").read_bytes()
        assert not (left / "grow-farm.plan.next").exists() and not (right / "grow-farm.plan.next").exists()
        if point == "initialized":
            for path in paths:
                verify(path)
    print("PASS staged-job recovery and unfinished partial recovery preserve the saved extent", flush=True)


def check_replot(binary, root, reference=None):
    # Protocol fixtures exercise our gates and recovery, not third-party
    # harvester qualification or measured coexistence on a real farm.
    directory = root / "migration"
    directory.mkdir()
    counter = 0

    def run(*args, code=0, point=None):
        nonlocal counter
        env = dict(os.environ)
        env.pop("XCHPLOT2_GROW_FAILPOINT", None)
        if point:
            env["XCHPLOT2_GROW_FAILPOINT"] = point
        result = subprocess.run([str(binary), str(args[0]), "--config", os.devnull, *map(str, args[1:])],
                                capture_output=True, text=True, encoding="utf-8", env=env, timeout=180)
        counter += 1
        (directory / f"{counter:03d}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert SEED not in result.stdout + result.stderr, "migration diagnostics exposed the private seed"
        assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)
        return result

    farm = directory / "old and new plots"
    farm.mkdir()
    reader = directory / "reader"
    (reader / "config").mkdir(parents=True)
    (reader / "config" / "config.yaml").write_text("self_hostname: localhost\nharvester:\n  num_threads: 1\n", encoding="utf-8")
    executable = directory / "chia-fixture"
    executable.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " +
                          shlex.quote(str(Path(__file__).resolve())) + ' --replot-rpc-fixture "$@"\n', encoding="utf-8")
    executable.chmod(0o700)
    fixture = {"roots": [str(farm)]}

    def flags(**values):
        (reader / "fixture.json").write_text(json.dumps({**fixture, **values}), encoding="utf-8")

    flags()
    source = farm / "legacy.plot"
    plot_id = hashlib.sha256(bytes.fromhex(POOL + FARMER)).digest()
    legacy = (b"Proof of Space Plot" + plot_id + bytes([18]) + struct.pack(">H", 4) + b"v1.0" +
              struct.pack(">H", 112) + bytes.fromhex(POOL + FARMER + "9a" * 32))
    first_table = len(legacy) + 80
    legacy += b"".join(struct.pack(">Q", first_table + 8 * i) for i in range(10)) + bytes(128)
    source.write_bytes(legacy)
    source_hash = digest(source)
    adapter = ["--chia-root", reader, "--chia-executable", executable]
    inventory_file = directory / "inventory.plan"
    catalog = json.loads(run("plan", "--inventory", farm, *adapter, "--save-plan", inventory_file, "--json").stdout)
    assert catalog["network"]["available"] and len(catalog["plots"]) == 1
    assert catalog["plots"][0]["header_validated"] and catalog["plots"][0]["plot_id"] == plot_id.hex()

    profile_file = directory / "mixed farm.json"
    profile = json.loads(run("plan", "--farming-cores", 1, "--farming-memory-bytes", "256M",
                             "--save-profile", profile_file, "--json").stdout)["farming_profile"]
    resources = catalog["farm_resources"]
    mib, ram = 1024 * 1024, 2 * 1024 * 1024 * 1024
    phase_cpu = min(2, len(os.sched_getaffinity(0)))
    cpu = next(r["id"] for r in resources["resources"] if r["kind"] == "cpu")
    memory = next(r["id"] for r in resources["resources"] if r["kind"] == "host_memory")
    disks = [r["id"] for r in resources["resources"] if r["kind"] == "storage"]
    for resource in resources["resources"]:
        resource.update(capacity=4 * ram if resource["kind"] == "host_memory" else phase_cpu if resource["kind"] == "cpu" else 1,
                        capacity_known=True, conditional_shared=resource["kind"] != "host_memory")
    resources.update(source="synthetic-protocol-fixture", producer="group-cli.py",
                     gate={"max_age_ms": 5000, "recovery_hold_ms": 0, "recovery_samples": 1,
                           "max_cpu_busy_permille": 950})
    for workload in resources["workloads"]:
        workload.update(format="chiapos-v1", decoder="fixture-reader", decoder_version="fixture-chia-v2",
                        backend="cpu", pool_difficulty="fixture", source="synthetic-protocol-fixture",
                        cost="serialized", bounded=True, scope_complete=True, quality_burst=1, proof_burst=1,
                        future_quality_burst=2, future_proof_burst=2, period_us=100000000, deadline_us=60000000,
                        uses=[{"resource": resource, "required_units": 1, "quality_us": 100, "proof_us": 100,
                               "blocking_us": 100} for resource in [cpu, *disks]] +
                             [{"resource": memory, "resident_units": mib, "transient_units": mib}])

    def demands(bytes_, cpu_lanes=1):
        return [{"resource": resource, "required_units": cpu_lanes if resource == cpu else 1, "work_us": 100, "blocking_us": 100,
                 "period_us": 100000000, "deadline_us": 60000000, "bounded": True} for resource in [cpu, *disks]] + [
                    {"resource": memory, "transient_units": bytes_, "bounded": True}]

    resources["native"] = {"bounded": True, "scope_complete": True, "covers_future_filter": True,
                           "phase_totals_include_controller": True, "host": catalog["host"], "k": 18, "strength": 2,
                           "reader": "stateless", "backend": "cpu", "engine": "incremental",
                           "max_ram_bytes": ram, "max_cpu_threads": 1, "max_batch_members": 1, "max_group_members": 1,
                           "max_construction_us": 120000000, "max_verification_us": 120000000,
                           "verification_cpu_threads": 1, "output_resources": disks, "scratch_resources": [],
                           "source_resources": disks, "construction": demands(ram + 64 * mib, phase_cpu),
                           "verification": demands(ram + 64 * mib, phase_cpu), "supervision": demands(ram, phase_cpu),
                           "cohorts": [{"max_members": 2, "new_farming": demands(mib), "proof_reserve": demands(mib)}]}
    profile["farm_resources"] = resources
    profile_file.write_text(json.dumps(profile), encoding="utf-8")
    options = ["-k", 18, "--group-members", 1, "--max-total-members", 1, "--max-groups", 1,
               "--workers", 1, "--batch-members", 1, "--construction", "incremental", "--destination-temp", "off",
               "--devices", "cpu", "--compute", "off", "-M", ram, "--headroom-bytes", 0,
               "--capacity-bytes", ram + 256 * mib, "--window-bytes", "64K", "--checkpoint-bytes", "64K"]
    keys = ["-f", FARMER, "--pool-ph", POOL, "-S", SEED]
    external_profile = json.loads(json.dumps(profile))
    external_profile["farm_resources"]["native"].update(engine="external", scratch_resources=disks)
    external_file = directory / "external profile.json"
    external_file.write_text(json.dumps(external_profile), encoding="utf-8")
    external_options = list(options)
    external_options[external_options.index("--construction") + 1] = "external"
    external_options[external_options.index("--destination-temp") + 1] = "on"
    guarded = snapshot(farm)
    refused = json.loads(run("plan", farm, "--replot", inventory_file, "--harvester", external_file,
                             *adapter, *external_options, "--reclaim-before-bytes", 1, "--json", code=2).stdout)
    assert any("external migration with positive pre-delete credit" in reason for reason in refused["infeasibility_reasons"])
    rejected_review = directory / "unsupported external.plan"
    run("plan", farm, "--replot", inventory_file, "--harvester", external_file, *adapter,
        *external_options, "--reclaim-before-bytes", 1, "--save-plan", rejected_review, code=2)
    assert not rejected_review.exists() and snapshot(farm) == guarded
    external_review = directory / "zero-credit external.plan"
    zero = json.loads(run("plan", farm, "--replot", inventory_file, "--harvester", external_file,
                          *adapter, *external_options, "--save-plan", external_review, "--json").stdout)
    assert zero["feasible"] and zero["reclaim_before_bytes"] == 0
    record = external_review.read_bytes()
    body = record[:-65]
    assert record[-65:] == hashlib.sha256(body).hexdigest().encode() + b"\n"
    rows = farming_plan_rows(body)
    assert rows[0] == b"ordinary-group-replot-review-v1\n" and rows[4].startswith(b"0 ")
    rows[4] = b"1 " + rows[4].split(b" ", 1)[1]
    body = b"".join(rows)
    rejected_review.write_bytes(body + hashlib.sha256(body).hexdigest().encode() + b"\n")
    imported = json.loads(run("plan", rejected_review, "--harvester", external_file,
                              *adapter, *external_options, "--json", code=2).stdout)
    assert any("external migration with positive pre-delete credit" in reason for reason in imported["infeasibility_reasons"])
    run("fill", rejected_review, "--harvester", external_file, *adapter, *external_options, *keys, code=2)
    assert snapshot(farm) == guarded, "imported external predelete review changed the farm"
    temporary_profile = json.loads(json.dumps(profile))
    temporary_profile["farm_resources"]["native"]["scratch_resources"] = disks
    temporary_file = directory / "temporary profile.json"
    temporary_file.write_text(json.dumps(temporary_profile), encoding="utf-8")
    valid_temp, invalid_temp = directory / "temporary directory", directory / "temporary regular file"
    valid_temp.mkdir()
    invalid_temp.write_text("not a directory", encoding="utf-8")
    bad_temp_review = directory / "invalid temporary.plan"
    invalid = json.loads(run("plan", farm, "--replot", inventory_file, "--harvester", temporary_file,
                             *adapter, *options, "--temp-dir", invalid_temp, "--reclaim-before-bytes", 1,
                             "--json", code=2).stdout)
    assert any("cannot inspect scratch/destination directory" in reason for reason in invalid["infeasibility_reasons"])
    run("plan", farm, "--replot", inventory_file, "--harvester", temporary_file, *adapter, *options,
        "--temp-dir", invalid_temp, "--reclaim-before-bytes", 1, "--save-plan", bad_temp_review, code=2)
    assert not bad_temp_review.exists() and snapshot(farm) == guarded
    temporary_review = directory / "valid temporary.plan"
    valid = json.loads(run("plan", farm, "--replot", inventory_file, "--harvester", temporary_file,
                           *adapter, *options, "--temp-dir", valid_temp, "--reclaim-before-bytes", 1,
                           "--save-plan", temporary_review, "--json").stdout)
    assert valid["feasible"] and valid["native_request"]["scratch_resources"] == disks
    imported = json.loads(run("plan", temporary_review, "--harvester", temporary_file, *adapter, *options,
                              "--temp-dir", invalid_temp, "--json", code=2).stdout)
    assert any("cannot inspect scratch/destination directory" in reason for reason in imported["infeasibility_reasons"])
    run("fill", temporary_review, "--harvester", temporary_file, *adapter, *options, "--temp-dir", invalid_temp, *keys, code=2)
    assert snapshot(farm) == guarded, "invalid temporary root changed the farm before predelete admission"
    review_file = directory / "migration.plan"
    before = snapshot(farm)
    review = json.loads(run("plan", farm, "--replot", inventory_file, "--harvester", profile_file,
                            *adapter, *options, "--save-plan", review_file, "--json").stdout)
    assert review["replot_review"] and review["feasible"] and not review["live_readiness_checked"]
    assert review["reclaim_before_bytes"] == 0 and review["farming_value_preserved"] is None
    assert len(review["sources"]) == len(review["batches"]) == 1 and snapshot(farm) == before
    final = Path(review["batches"][0]["path"])
    assert final.suffix == ".plot2" and review["job_id"] in final.name
    assert SEED not in review_file.read_text(encoding="utf-8")
    saved_review = review_file.read_bytes(), review_file.stat().st_mtime_ns
    execute = ["fill", review_file, "--harvester", profile_file, *adapter, *options]
    for failure in ("farmer_unavailable", "solver_unavailable", "farmer_disconnected", "solver_stopped", "pool_external", "pool_mismatch", "node_mismatch"):
        flags(**{failure: True})
        unavailable = run(*execute, *keys, code=2)
        assert snapshot(farm) == before and not (farm / "grow-replot.plan").exists()
        if failure == "pool_external":
            assert "external pooled target execution is unsupported" in unavailable.stdout + unavailable.stderr
    flags()
    for name in ("farmer_version", "solver_version"):
        flags(**{name: "changed-service-version"})
        drift = run(*execute, *keys, code=2)
        assert "service version changed since review" in drift.stdout + drift.stderr and snapshot(farm) == before
    flags()
    og = run(*execute, "-f", FARMER, "--pool-pk", FARMER, "-S", SEED, code=2)
    assert "OG pool-public-key target execution is unsupported" in og.stdout + og.stderr and snapshot(farm) == before
    run(*execute, *keys, "--compute", "auto", code=2)
    assert snapshot(farm) == before, "uncovered compute changed the farm"
    run(*execute, *keys, point="replot-journal-all-staged", code=86)
    staged_private = snapshot(farm)
    for failure in ("farmer_disconnected", "solver_stopped"):
        flags(**{failure: True})
        run(*execute, code=2)
        assert snapshot(farm) == staged_private, "keyless service refusal changed an interrupted private journal or source"
    flags()
    run(*execute, *keys, point="replot-journal-one-published", code=86)
    assert digest(source) == source_hash and not (farm / "grow-farm.plan").exists()
    run("fill", farm, "--resume", "--compute", "off", "--devices", "cpu", code=2)
    guarded = snapshot(farm)
    named = run("plot", "-k", 18, "-n", 1, *keys, "-o", farm, "--group-members", 1,
                "--devices", "cpu", "--compute", "off", "-M", ram, code=2)
    assert "migration-managed directory" in named.stdout + named.stderr
    named = run("plot", "-k", 18, "-n", 1, *keys, "-o", directory / "unmanaged output",
                "--manifest", farm / "intruding-job.tsv", "--group-members", 1,
                "--devices", "cpu", "--compute", "off", "-M", ram, code=2)
    assert "migration-managed directory" in named.stdout + named.stderr
    manifest = directory / "guarded batch.tsv"
    manifest.write_text("18 2 0 0 gplot-v2-grouped " + "12" * 32 + " " +
                        (POOL + FARMER + "9a" * 32) + " " + json.dumps(str(farm), ensure_ascii=False) +
                        ' "ordinary.gplot" 1\n', encoding="utf-8")
    batched = run("batch", manifest, "--devices", "cpu", "--compute", "off", "-M", ram, code=2)
    assert "migration-managed directory" in batched.stdout + batched.stderr
    rollover_reference = reference.with_name("pos2_pr118_grow") if reference else None
    if rollover_reference and rollover_reference.is_file():
        rolled = subprocess.run([str(rollover_reference), str(farm), "--resume", "-k", "18", *keys,
                                 "--group-members", "1", "--max-ram", str(ram)], capture_output=True,
                                text=True, encoding="utf-8", timeout=5)
        (directory / "guarded-rollover.log").write_text(rolled.stdout + rolled.stderr, encoding="utf-8")
        assert rolled.returncode == 1 and "migration-managed directory" in rolled.stdout + rolled.stderr
    assert snapshot(farm) == guarded, "another group entrypoint changed the active migration"
    run(*execute, point="replot-replacement-verified", code=86)
    assert digest(source) == source_hash and not final.exists()
    staged, = farm.glob(".xchplot2-stage-*.building")
    guarded = snapshot(farm)
    direct = run("grow", staged, "--max-members", 2, "--devices", "cpu", "--compute", "off", "-M", ram, code=2)
    assert "migration-managed directory" in direct.stdout + direct.stderr
    assert snapshot(farm) == guarded, "direct growth changed an active migration's sealed stage"
    run(*execute, point="replot-replacement-published", code=86)
    replacement_hash = digest(final)
    guarded = snapshot(farm)
    immutable = run("grow", final, "--max-members", 2, "--devices", "cpu", "--compute", "off", "-M", ram, code=2)
    assert "published migration groups are immutable" in immutable.stdout + immutable.stderr
    assert snapshot(farm) == guarded, "direct growth changed an immutable migration publication"

    # Each refusal occurs with a real sealed replacement present. Saved review
    # bytes cannot bypass fresh network and exact loaded-key observations.
    for failure in ("unloaded", "wrong_key", "inactive", "stale", "farmer_disconnected", "farmer_unloaded",
                    "farmer_syncing", "farmer_mismatch", "solver_stopped", "solver_disconnected", "pool_external"):
        flags(**{failure: True})
        run(*execute, code=2)
        assert digest(source) == source_hash and digest(final) == replacement_hash
        assert not (farm / "grow-farm.plan").exists()
    flags()
    run(*execute, point="replot-replacement-ready", code=86)
    run(*execute, point="replot-retirement-intent", code=86)
    assert digest(source) == source_hash
    flags(farmer_disconnected=True)
    run(*execute, code=2)
    assert source.exists() and digest(source) == source_hash and digest(final) == replacement_hash
    flags(inactive=True)
    run(*execute, code=2)
    assert source.exists() and digest(source) == source_hash, "resumed intent hid the source without fresh readiness"
    flags()
    run(*execute, point="replot-source-renamed", code=86)
    assert not source.exists()
    run(*execute, point="replot-retirement-claimed", code=86)
    run(*execute, point="replot-before-unlink", code=86)
    claims = list(farm.glob(".xchplot2-retire-*/**/*"))
    assert any(p.is_file() and digest(p) == source_hash for p in claims), "claim disappeared before unlink"
    flags(solver_stopped=True)
    run(*execute, code=2)
    assert any(p.is_file() and digest(p) == source_hash for p in claims), "stopped solver removed a claimed source"
    flags(inactive=True)
    run(*execute, code=2)
    assert any(p.is_file() and digest(p) == source_hash for p in claims), "network refusal removed the claimed source"
    flags()
    for point in ("replot-source-unlinked", "replot-receipt-synced", "replot-handoff-all-staged",
                  "replot-handoff-one-published", "replot-handoff-complete"):
        run(*execute, point=point, code=86)
        assert digest(final) == replacement_hash
    run(*execute)
    assert not source.exists() and (farm / "grow-farm.plan").is_file() and (farm / "grow-replot.plan").is_file()
    assert (review_file.read_bytes(), review_file.stat().st_mtime_ns) == saved_review
    for name in ("grow-farm.plan", "grow-replot.plan"):
        assert (farm / name).stat().st_mode & 0o777 == 0o600
    proof = run("verify", final, "--full", "--trials", 100)
    found = re.search(r"(\d+) full proofs validated", (proof.stdout + proof.stderr).lower())
    assert found and int(found[1]) > 0
    assert header(final)[3] == 1
    run("fill", farm, "--resume", "--devices", "cpu", "--compute", "off", "-M", ram, "--quiet")
    assert digest(final) == replacement_hash and list(farm.glob("*.plot2")) == [final]
    if reference:
        group, k, strength, count, meta, memo = header(final)
        fragments = directory / "reference fragments"
        fragments.mkdir()
        raw = fragments / "0.plot2"
        run("test", k, group, strength, 0, meta, "--raw", "-m", memo, "-o", fragments, "-N", raw.name)
        inputs = fragments / "group.inputs"
        inputs.write_text(f"{k} {strength} {meta} {group} {memo}\n" + json.dumps(str(raw), ensure_ascii=False) + "\n", encoding="utf-8")
        canonical = fragments / "canonical.gplot"
        comparison = subprocess.run([str(reference), "assemble-only", str(inputs), str(canonical), "512"],
                                    capture_output=True, text=True, encoding="utf-8", timeout=180)
        assert comparison.returncode == 0, (comparison.stdout, comparison.stderr)
        assert digest(canonical) == replacement_hash
    print("PASS reviewed migration, fresh readiness refusals, crash recovery, immutable handoff and full proofs", flush=True)

    # Source and target implementations use distinct explicit local routes.
    # The old route deliberately has no compatible target full-node interface.
    routed_farm = directory / "routed farm"
    routed_farm.mkdir()
    routed_source = routed_farm / "legacy.plot"
    routed_source.write_bytes(legacy)
    route_specs = []
    for name, kind, extra in (("old-gh", "gigahorse", {"source_only": True, "version": "fixture-gh-v1"}),
                              ("new-chia", "chia", {"target_only": True})):
        config = directory / name
        (config / "config").mkdir(parents=True)
        (config / "config" / "config.yaml").write_text("self_hostname: localhost\nharvester:\n  num_threads: 1\n", encoding="utf-8")
        (config / "fixture.json").write_text(json.dumps({"roots": [str(routed_farm)], **extra}), encoding="utf-8")
        route_specs.append({"id": name, "kind": kind, "root": str(config), "executable": str(executable),
                            "source_roots": [str(routed_farm)] if name == "old-gh" else []})
    routed_profile = json.loads(json.dumps(profile))
    routed_profile["farm_adapters"] = {"schema": 1, "target_route": "new-chia", "routes": route_specs}
    routed_file = directory / "routed profile.json"
    routed_file.write_text(json.dumps(routed_profile), encoding="utf-8")
    routed_inventory = directory / "routed inventory.plan"
    observed = json.loads(run("plan", "--inventory", routed_farm, "--harvester", routed_file,
                              "--save-plan", routed_inventory, "--json").stdout)
    assert observed["plots"][0]["harvester"] == "old-gh" and observed["network"]["target_route"] == "new-chia"
    routed_review = directory / "routed migration.plan"
    reviewed = json.loads(run("plan", routed_farm, "--replot", routed_inventory, "--harvester", routed_file,
                              *options, "--save-plan", routed_review, "--json").stdout)
    before = snapshot(routed_farm)
    run("fill", routed_review, *keys, *adapter, *options, code=2)
    assert snapshot(routed_farm) == before, "saved public route IDs authorized a missing local adapter profile"
    swapped = json.loads(json.dumps(routed_profile))
    swapped["farm_adapters"]["target_route"] = "old-gh"
    wrong_route = directory / "wrong route.json"
    wrong_route.write_text(json.dumps(swapped), encoding="utf-8")
    run("fill", routed_review, *keys, "--harvester", wrong_route, *options, code=2)
    assert snapshot(routed_farm) == before, "changed target route wrote before rejecting the saved binding"
    uncovered = json.loads(json.dumps(routed_profile))
    uncovered["farm_adapters"]["routes"][0]["source_roots"] = []
    wrong_route.write_text(json.dumps(uncovered), encoding="utf-8")
    run("fill", routed_review, *keys, "--harvester", wrong_route, *options, code=2)
    assert snapshot(routed_farm) == before, "missing source route wrote before rejection"
    run("fill", routed_review, *keys, "--harvester", routed_file, *options,
        point="replot-replacement-published", code=86)
    routed_final = Path(reviewed["batches"][0]["path"])
    routed_hash = digest(routed_final)
    old_fixture = Path(route_specs[0]["root"]) / "fixture.json"
    original_fixture = old_fixture.read_bytes()
    try:
        old_fixture.write_text(json.dumps({"roots": [str(routed_farm)], "version": "fixture-gh-v1"}), encoding="utf-8")
        doubled = run("fill", routed_review, "--harvester", routed_file, *options, code=2)
        assert "outside the reviewed reader/catalog scope" in doubled.stdout + doubled.stderr
        assert routed_source.exists() and digest(routed_final) == routed_hash
    finally:
        old_fixture.write_bytes(original_fixture)
    run("fill", routed_review, "--harvester", routed_file, *options)
    routed_final = Path(reviewed["batches"][0]["path"])
    assert not routed_source.exists() and digest(routed_final) == replacement_hash
    assert (routed_farm / "grow-farm.plan").is_file()
    print("PASS distinct source/target adapters and missing/swapped route rejection before writes", flush=True)


def check_compute(binary, root):
    # Reuse the established TLS and process lifecycle fixtures.
    import importlib.util
    spec = importlib.util.spec_from_file_location("group_pool_test", Path(__file__).with_name("group-pool-test.py"))
    pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pool)
    ca, identities = pool.certificates(root / "tls")
    broker_port, worker_port = pool.port(), pool.port()
    endpoint, worker_endpoint = f"127.0.0.1:{broker_port}", f"127.0.0.1:{worker_port}"
    processes = []

    def start(args, name):
        process = pool.Process([binary, *args, "--config", os.devnull], root / (name + ".log"))
        processes.append(process)
        return process

    try:
        cert, key = identities["broker"]
        broker = start(["compute", "broker", "--listen", endpoint, "--tls-ca", ca,
                        "--tls-cert", cert, "--tls-key", key], "broker")
        pool.wait_listener(broker, broker_port)
        cert, key = identities["worker-a"]
        worker = start(["compute", "worker", "--broker", endpoint, "--listen", worker_endpoint,
                        "--advertise", worker_endpoint, "--ca", ca, "--cert", cert, "--key", key,
                        "--cpus", 2, "--slots", 2, "--host-bytes", 1024 * 1024 * 1024,
                        "--generation-threads", 1, "--backend", "cpu"], "worker")
        pool.wait_listener(worker, worker_port)
        common = ["plot", "-k", 18, "-n", 1, "-f", FARMER, "--pool-ph", POOL,
                  "--group-members", 3, "--batch-members", 2, "--backend", "cpu",
                  "--construction", "staged", "--max-host-ram", "512M", "--workers", 2,
                  "--window-bytes", "64K", "--checkpoint-bytes", "64K", "--quiet"]
        owners = []
        for owner in ("owner-a", "owner-b"):
            cert, key = identities[owner]
            network = ["--compute-broker", endpoint, "--compute-owner-id", owner,
                       "--compute-ca", ca, "--compute-cert", cert, "--compute-key", key,
                       "--compute-inflight", 2, "--compute-codec-ram", 0, "--compute-timeout-ms", 120000]
            owners.append((owner, start([*common, "-S", SEED, "-o", root / owner, *network], owner)))
        paths = []
        for owner, process in owners:
            path = Path(process.finish().strip())
            assert path.is_file() and header(path)[3] == 3
            paths.append(path)
            scoped = hashlib.sha256(b"xchplot2-pool-storage-owner-v1" + bytes.fromhex(SEED) + owner.encode()).hexdigest()
            result = pool.run([binary, *common, "-S", scoped, "-o", root / (owner + "-local"), "--config", os.devnull])
            local = Path(result.stdout.strip())
            assert digest(local) == digest(path), "remote group differs from local construction"
            result = pool.run([binary, "verify", path, "--full", "--trials", 100, "--config", os.devnull])
            proof = re.search(r"(\d+) full proofs validated", (result.stdout + result.stderr).lower())
            assert proof and int(proof[1]) > 0
        assert header(paths[0])[0] != header(paths[1])[0], "owners reused a group identity"
        print("PASS two simultaneous plotting owners share one compute worker, byte parity and full proofs", flush=True)
    finally:
        for process in reversed(processes):
            process.stop()


def check_automatic_compute(binary, root):
    """Saved membership, embedded brokers, discovery and shared executors."""
    import socket

    directory = root / "automatic-compute"
    directory.mkdir()
    states = {name: directory / name for name in ("first", "second", "worker")}
    processes = []
    counter = 0

    def command(*args):
        return [str(binary), *map(str, args), "--config", os.devnull]

    def environment(name):
        env = dict(os.environ)
        env["XDG_CONFIG_HOME"] = str(states[name])
        env.pop("XCHPLOT2_GROW_FAILPOINT", None)
        return env

    def run(name, *args, code=0):
        nonlocal counter
        result = subprocess.run(command(*args), env=environment(name), capture_output=True, text=True, encoding="utf-8", timeout=180)
        counter += 1
        (directory / f"{counter:03d}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)
        assert "PRIVATE KEY" not in result.stdout + result.stderr
        return result

    def start(name, label, *args):
        path = directory / (label + ".log")
        stream = path.open("w")
        process = subprocess.Popen(command(*args), env=environment(name), stdout=stream, stderr=subprocess.STDOUT)
        processes.append((process, stream, path))
        return process, path

    def finish(process, path, service=None):
        deadline = time.monotonic() + 180
        while process.poll() is None:
            if service:
                required, log = service
                assert required.poll() is None, log.read_text(encoding="utf-8")
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(process.args, 180)
            time.sleep(0.1)
        result = process.returncode
        output = path.read_text(encoding="utf-8")
        assert result == 0, (result, output)
        paths = [Path(line) for line in output.splitlines() if line.endswith(".gplot")]
        assert paths and all(p.is_file() for p in paths), output
        return paths

    def stop(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    keys = ["-f", FARMER, "--pool-ph", POOL]
    limits = ["--devices", "cpu", "--max-host-ram", "1G", "--workers", 2,
              "--window-bytes", "64K", "--checkpoint-bytes", "64K",
              "--compute-timeout-ms", 60000, "--quiet"]
    plot = ["plot", "-k", 18, *keys, *limits]
    try:
        run("first", "compute", "init", "--name", "storage A")
        for name in ("second", "worker"):
            invitation = directory / (name + ".invite")
            run("first", "compute", "invite", invitation, "--name", name)
            run(name, "compute", "join", invitation)
            if os.name != "nt":
                assert invitation.stat().st_mode & 0o777 == 0o600
        for name in states:
            run(name, "compute", "configure", "--interface", "lo", "--backend", "cpu",
                "--cpus", 1, "--host-bytes", "512M")
        refused = run("worker", "compute", "worker", "--max-brokers", 128, code=1)
        assert "reduce --max-brokers or increase --host-bytes" in refused.stderr, refused
        run("worker", "compute", "configure", "--max-brokers", 16)
        assert "automatic for each plotting job" in run("second", "compute", "status").stdout
        preview = directory / "preview"
        preview.mkdir()
        before = {name: snapshot(state) for name, state in states.items()}
        automatic_profile = directory / "harvester.json"
        exported = run("first", "plan", "--save-profile", automatic_profile, "-M", "16M", "--quiet")
        assert automatic_profile.is_file() and not exported.stdout and not exported.stderr
        assert all(snapshot(states[name]) == saved for name, saved in before.items()), "profile export changed compute settings"
        plan_options = [preview, "-k", 18, "--group-members", 2, "--max-total-members", 3,
                        "--capacity-bytes", "64M", "--headroom-bytes", 0, *limits, "--json"]
        plan = run("first", "plan", *plan_options)
        dry_fill = run("first", "fill", *plan_options, "--plan-only")
        assert json.loads(plan.stdout) == json.loads(dry_fill.stdout)
        assert not list(preview.iterdir()) and all(snapshot(states[name]) == saved for name, saved in before.items())
        automatic_review = directory / "automatic.plan"
        run("first", "plan", *plan_options, "--save-plan", automatic_review)
        assert automatic_review.is_file() and not list(preview.iterdir())
        assert all(snapshot(states[name]) == saved for name, saved in before.items()), "saved review started compute services"

        # No worker or dedicated broker is running: a normal plot still works.
        made = run("first", *plot, "--max-host-ram", "512M", "-S", SEED, "-o", directory / "local")
        output = Path(made.stdout.strip())
        assert output.is_file() and header(output)[3] == 1 and not made.stderr
        profile = (states["first"] / "xchplot2/compute/profile").read_text(encoding="utf-8").splitlines()
        node = shlex.split(profile[1])[1]
        scoped = hashlib.sha256(b"xchplot2-pool-storage-owner-v1" + bytes.fromhex(SEED) + node.encode()).hexdigest()
        local = Path(run("first", *plot, "--compute", "off", "-S", scoped, "-o", directory / "reference").stdout.strip())
        assert digest(local) == digest(output), "automatic local executor changed plot bytes"
        run("first", "grow", output, "--max-members", 3, *limits)
        assert header(output)[3] == 3
        run("first", "verify", output, "--full", "--trials", 100)
        filled = run("first", "fill", "--from-plan", automatic_review, *keys, "-S", SEED, *limits)
        fill_paths = [Path(line) for line in filled.stdout.splitlines() if line.endswith(".gplot")]
        assert len(fill_paths) == 2 and sum(header(path)[3] for path in fill_paths) == 3, filled
        print("PASS saved enrollment, read-only plans and automatic local plot/fill/grow", flush=True)

        # Both storage machines broker their own work; a worker can arrive later.
        jobs = [start(name, name + "-remote", *plot, "--group-members", 3, "-S", SEED,
                      "--compute-local", "off", "-o", directory / (name + "-remote"))
                for name in ("first", "second")]
        time.sleep(0.5)
        assert all(process.poll() is None for process, _ in jobs), [path.read_text(encoding="utf-8") for _, path in jobs]
        worker, worker_log = start("worker", "shared-worker", "compute", "worker")
        remote_paths = [finish(process, path, (worker, worker_log))[0] for process, path in jobs]
        assert all(header(path)[3] == 3 for path in remote_paths)
        assert header(remote_paths[0])[0] != header(remote_paths[1])[0]
        assert worker.poll() is None, worker_log.read_text(encoding="utf-8")
        for path in remote_paths:
            run("first", "verify", path, "--full", "--trials", 100)
        print("PASS worker arrival and one worker shared by two automatic brokers", flush=True)

        # A saved dedicated broker replaces the per-job broker.
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        broker, broker_log = start("first", "dedicated-broker", "compute", "broker", "--listen", f"127.0.0.1:{port}",
                                   "--worker-ms", 3000, "--lease-ms", 3000)
        run("second", "compute", "configure", "--broker", f"127.0.0.1:{port}")
        result = run("second", *plot, "--compute-local", "off", "-o", directory / "dedicated", "--no-quiet", "-v")
        assert "Using broker 127.0.0.1:" in result.stderr and "Automatic broker" not in result.stderr
        assert Path(result.stdout.strip()).is_file()
        assert broker.poll() is None, broker_log.read_text(encoding="utf-8")
        explicit = [*plot, "--compute-owner-id", "storage-test", "--compute-local", "off",
                    "-S", SEED, "-o", directory / "explicit-owner"]
        owned = Path(run("second", *explicit).stdout.strip())
        before = digest(owned)
        assert Path(run("second", *explicit, "--resume").stdout.strip()) == owned
        assert digest(owned) == before
        scoped_owner = hashlib.sha256(b"xchplot2-pool-storage-owner-v1" + bytes.fromhex(SEED) + b"storage-test").hexdigest()
        reference = Path(run("second", *plot, "--compute", "off", "-S", scoped_owner,
                             "-o", directory / "explicit-reference").stdout.strip())
        assert digest(reference) == before, "explicit owner namespace was not preserved"
        # Kill a frozen worker while the owner is waiting on its lease. A new
        # process with the same enrolled identity takes over after expiration.
        if os.name == "nt":
            worker.kill()
            worker.wait(timeout=10)
        else:
            worker.send_signal(signal.SIGSTOP)
        recovering, recovering_log = start("second", "worker-restart", *plot, "--group-members", 3, "-S", SEED,
                                            "--compute-local", "off", "--compute-socket-timeout-ms", 1000,
                                            "-o", directory / "worker-restart")
        time.sleep(0.3)
        assert recovering.poll() is None, recovering_log.read_text(encoding="utf-8")
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)
        worker, worker_log = start("worker", "restarted-worker", "compute", "worker")
        recovered = finish(recovering, recovering_log, (worker, worker_log))[0]
        assert digest(recovered) == digest(remote_paths[1]), "worker loss changed the recovered group"
        stop(worker)
        stop(broker)
        # A dedicated broker outage still permits the configured local
        # executor to finish through the existing bounded fallback.
        fallback = run("second", *plot, "--compute-socket-timeout-ms", 1000,
                       "-o", directory / "broker-outage")
        run("second", "verify", Path(fallback.stdout.strip()), "--full", "--trials", 100)
        run("second", "compute", "configure", "--broker", "auto")
        print("PASS worker replacement and local progress during dedicated broker outage", flush=True)

        # Two concurrent jobs on the same enrolled machine reuse one executor.
        same_host = [start("first", f"same-host-{i}", *plot, "--group-members", 3, "-n", 2,
                           "-o", directory / f"same-host-{i}") for i in range(2)]
        for process, path in same_host:
            assert len(finish(process, path)) == 2
        print("PASS dedicated broker selection and concurrent local plotting jobs", flush=True)
    finally:
        for process, stream, _ in reversed(processes):
            stop(process)
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--compute", action="store_true", help="also test authenticated loopback compute sharing (needs openssl)")
    parser.add_argument("--automatic-compute", action="store_true", help="test saved enrollment and automatic LAN compute over loopback")
    parser.add_argument("--scratch", type=Path, default=Path.cwd(), help="existing directory on real disk")
    args = parser.parse_args()
    allocated_console = False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        allocated_console = kernel.AllocConsole()
        if not allocated_console and ctypes.get_last_error() != 5:
            raise ctypes.WinError(ctypes.get_last_error())
    try:
        with tempfile.TemporaryDirectory(prefix="xchplot2 group é-", dir=args.scratch.resolve()) as work:
            os.environ["XDG_CONFIG_HOME"] = str(Path(work) / "empty-compute-settings")
            check(args.binary.resolve(), Path(work), args.reference.resolve() if args.reference else None)
            if sys.platform == "linux":
                check_replot(args.binary.resolve(), Path(work), args.reference.resolve() if args.reference else None)
            if args.compute:
                check_compute(args.binary.resolve(), Path(work))
            if args.automatic_compute:
                check_automatic_compute(args.binary.resolve(), Path(work))
    finally:
        if allocated_console:
            kernel.FreeConsole()



if __name__ == "__main__":
    if sys.argv[1:2] == ["--replot-rpc-fixture"]:
        replot_rpc_fixture(sys.argv[2:])
    else:
        main()
