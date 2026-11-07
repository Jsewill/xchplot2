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
# The same pinned derivation after compute_owner_seed for the fixture's
# protected-migration-owner namespace; both public keys are checked by group ID.
COMPUTE_FARMING_PLOT_KEY = "88485b2103714b3791ae725a4d3049de6cbf83fb6b56c5c788801fd5a4405a8e3b29e057024d59691cb755be061edd93"


def replot_rpc_fixture(args):
    """Typed local RPC fixture; no running farmer or network is contacted."""
    assert len(args) in (5, 6) and args[0] == "--root-path" and args[2] == "rpc", args
    root = Path(args[1])
    fixture = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
    service, endpoint = args[3:5]
    assert service in ("harvester", "full_node", "farmer", "solver"), service
    assert service == "harvester" or not fixture.get("source_only"), "source reader used as a target service"
    mutation = fixture.get("mutate_source")
    if mutation and not (root / "source-mutated").exists():
        source = Path(mutation["source"])
        claims = list(source.parent.glob(".xchplot2-retire-*/source-*"))
        published = list(source.parent.glob("*.plot2"))
        trigger = ((mutation["when"] == "published" and endpoint == "get_plots" and published) or
                   (mutation["when"] == "claimed" and endpoint == "get_harvesters_summary" and claims))
        if trigger:
            old_root = Path(mutation["root"])
            old_fixture = old_root / "fixture.json"
            old = json.loads(old_fixture.read_text(encoding="utf-8"))
            if mutation["change"] == "config":
                config = old_root / "config" / "config.yaml"
                text = config.read_text(encoding="utf-8")
                assert "num_threads: 1" in text
                config.write_text(text.replace("num_threads: 1", "num_threads: 2"), encoding="utf-8")
            elif mutation["change"] == "version":
                old["version"] = "fixture-gh-v2"
                old_fixture.write_text(json.dumps(old), encoding="utf-8")
            elif mutation["change"] == "target_population":
                # Native harvester discovery precedes completed farmer sync.
                # The farmer registry and its generation intentionally stay
                # unchanged until the controller's fresh native route fence.
                fixture["native_arrival"] = str(root / "unreviewed-native.plot2")
                (root / "fixture.json").write_text(json.dumps(fixture), encoding="utf-8")
            else:
                assert mutation["change"] == "population"
                arrival = old_root / "unreviewed-arrival"
                arrival.mkdir()
                (arrival / "new.plot").write_bytes((source if source.exists() else claims[0]).read_bytes())
                old["roots"].append(str(arrival))
                old_fixture.write_text(json.dumps(old), encoding="utf-8")
            (root / "source-mutated").write_text(mutation["change"], encoding="utf-8")
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
                                 "current_difficulty": None if fixture.get("pool_uninitialized") else fixture.get("pool_difficulty", 17),
                                 "authentication_token_timeout": 5,
                                 "pool_config": {"p2_singleton_puzzle_hash": pool,
                                                 "launcher_id": "11" * 32, "target_puzzle_hash": "22" * 32,
                                                 "owner_public_key": FARMER, "version": fixture.get("pool_protocol_version", 1),
                                                 "pool_url": "https://pool.example" if fixture.get("pool_external") else ""}}]}
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
                              "plot_public_key": fixture.get("legacy_plot_public_keys", {}).get(str(path), FARMER)})
            if fixture.get("unloaded") or fixture.get("source_only"):
                continue
            for path in sorted(directory.glob("*.plot2")):
                group, _, strength, members, meta, memo = header(path)
                pooled = len(memo) == 224
                assert pooled or len(memo) == 256
                route_bytes = 64 if pooled else 96
                route = memo[:route_bytes]
                assert route == (POOL if pooled else FARMER) and memo[route_bytes:route_bytes + 96] == FARMER
                plot_key = fixture.get("plot_public_key", FARMING_PLOT_KEY)
                expected = hashlib.sha256(bytes([strength]) + bytes.fromhex(plot_key + route)).hexdigest()
                assert group == expected, "fixture's public key no longer matches the seeded family"
                stat = path.stat()
                plots.append({"filename": str(path), "plot_id": group, "size": 0x80 | strength,
                              "file_size": stat.st_size, "time_modified": int(stat.st_mtime),
                              "pool_contract_puzzle_hash": route if pooled else None, "pool_public_key": None if pooled else route,
                              "plot_public_key": FARMER if fixture.get("wrong_key") else plot_key,
                              "group_size": members, "meta_group": meta})
        if endpoint == "get_plots":
            if fixture.get("native_arrival"):
                plots.append({"filename": fixture["native_arrival"], "plot_id": "9f" * 32})
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
                         "--farm-adapter-timeout-ms", 120000,
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
    preparation = catalog["resource_preparation"]
    assert preparation["assessment_required"] and preparation["missing"]
    assert {fact["path"] for fact in preparation["facts"] if fact["basis"] == "observed"} >= {
        "local.logical_cpus", "local.available_memory_bytes"}
    assert any(item["path"] == "farm_resources.native.cohorts" for item in preparation["missing"])
    assert catalog["farm_resources"]["native"]["source_resources"]
    assert not catalog["farm_resources"]["native"]["output_resources"]
    assert all(workload["decoder"] == "unknown" for workload in catalog["farm_resources"]["workloads"])
    assert "9a" * 32 not in inventory_text and json.loads(inventory_file.read_text(encoding="utf-8")) == catalog
    # Resource-only inputs and whole inventories both reuse exact declarations;
    # diagnostic OS snapshots never replace capacities or cost flags.
    declared_resources = json.loads(json.dumps(catalog["farm_resources"]))
    declared_resources["source"] = "operator-declared-test-scope"
    declared_resources["resources"][0]["capacity"] = 7
    declared_resources["resources"][0]["capacity_known"] = True
    resource_file = root / "declared-resources.json"
    resource_file.write_text(json.dumps(declared_resources), encoding="utf-8")
    reused_file = root / "reused-inventory.plan"
    reused = json.loads(run("plan", "--inventory", inventory_root, "-M", "128M",
                            "--harvester", resource_file, "--save-plan", reused_file, "--json").stdout)
    assert reused["farm_resources"] == declared_resources
    assert any(fact["path"] == "farm_resources" and fact["basis"] == "declared"
               for fact in reused["resource_preparation"]["facts"])
    reused_again = json.loads(run("plan", "--inventory", inventory_root, "-M", "128M",
                                  "--harvester", reused_file, "--json").stdout)
    assert reused_again["farm_resources"] == declared_resources and reused_again["resource_preparation"]["missing"]
    saved_inventory = inventory_file.read_bytes(), inventory_file.stat().st_mtime_ns
    run("plan", "--inventory", inventory_root, "--save-plan", inventory_file, code=2)
    assert (inventory_file.read_bytes(), inventory_file.stat().st_mtime_ns) == saved_inventory
    assert not run("plan", "--inventory", inventory_root, "--quiet").stdout
    run("plan", "--inventory", code=2)
    run("fill", "--inventory", inventory_root, code=2)
    run("plan", "--inventory", inventory_root, "--metadata-only", code=2)
    for timeout in (0, 3600001):
        run("plan", "--inventory", inventory_root, "--farm-adapter-timeout-ms", timeout, code=2)
    run("plan", "--inventory", inventory_root, "--farm-adapter-timeout-ms", code=2)
    assert not run("plan", "--inventory", inventory_root, "--farm-adapter-timeout-ms", 3600000, "--quiet").stdout
    unrelated = run("plan", inventory_root, "--farm-adapter-timeout-ms", 120000, code=2)
    assert "requires inventory" in unrelated.stdout + unrelated.stderr
    run("grow", inventory_root / "legacy with spaces.plot", "--farm-adapter-timeout-ms", 120000, code=2)
    for mode in ("--status", "--metadata-only", "--calibrate"):
        run("plan", mode, inventory_file, "--farm-adapter-timeout-ms", 120000, code=2)
    for shell in ("bash", "zsh", "fish"):
        assert "farm-adapter-timeout-ms" in run("completions", shell).stdout
    unprojected = root / "unprojected-migration.plan"
    run("plan", inventory_root, "--replot", inventory_file, "--save-plan", unprojected,
        "--compute", "off", code=2)
    assert not unprojected.exists() and snapshot(inventory_root) == inventory_before
    print("PASS public farm inventory, observed preparation, exact declaration reuse and read-only migration rejection", flush=True)

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

    # The caller's large memory grant must survive public serialization too.
    large_root = directory / "large inventory"
    large_root.mkdir()
    for index in range(18000):
        (large_root / f"entry-{index:05d}").touch()
    large_before = snapshot(large_root)
    large_file = directory / "large inventory.plan"
    large = run("plan", "--inventory", large_root, "-M", "16G", "--save-plan", large_file, "--json")
    large_catalog = json.loads(large.stdout)
    assert len(large_catalog["plots"]) == 18000 and len(large_file.read_bytes()) > 16 * 1024 * 1024
    assert snapshot(large_root) == large_before, "large inventory serialization changed its source entries"

    def saved_replacement_authority(path, phase=None):
        record = path.read_bytes()
        body = record[:-65]
        assert record[-65:] == hashlib.sha256(body).hexdigest().encode() + b"\n"
        rows = farming_plan_rows(body)
        if phase is not None:
            assert rows[0] in (b"ordinary-group-replot-journal-v1\n", b"ordinary-group-replot-journal-v2\n",
                               b"ordinary-group-replot-journal-v3\n")
            if rows[0] != b"ordinary-group-replot-journal-v3\n":
                return "stamp"
            identities = dict(zip(("verified_stage", "published", "ready"), rows[-3:]))
            identity = identities[phase]
        else:
            assert rows[0] in (b"ordinary-group-replot-receipt-v1\n", b"ordinary-group-replot-receipt-v2\n")
            if rows[0] == b"ordinary-group-replot-receipt-v1\n":
                return "stamp"
            identity = rows[-1]
        scheme, filesystem, handle_type, object_id = shlex.split(identity.decode("utf-8"))
        assert scheme in ("0", "1", "2")
        if phase is None:
            assert scheme != "0", "native receipt omitted its replacement owner"
        return "stamp" if scheme == "0" else "owner"

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
    inventory_before = snapshot(farm)
    run("plan", "--inventory", farm, *adapter, "--farm-adapter-timeout-ms", 1, "--json", code=2)
    assert snapshot(farm) == inventory_before, "expired adapter inventory capture changed its source"
    assert snapshot(farm) == inventory_before and digest(source) == source_hash

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
    external_scratch = directory / "external migration scratch"
    external_scratch.mkdir()
    external_options += ["--temp-dir", external_scratch]
    guarded = snapshot(farm)
    external_review = directory / "positive-credit external.plan"
    positive_credit = catalog["plots"][0]["identity"]["allocated_bytes"]
    for name, declarations in (("raw", resources), ("wrapped", {"farm_resources": resources})):
        reusable_profile = directory / f"{name} resource declarations.json"
        reusable_profile.write_text(json.dumps(declarations), encoding="utf-8")
        reusable = json.loads(run("plan", farm, "--replot", inventory_file, "--harvester", reusable_profile,
                                  *adapter, *options, "--json").stdout)
        assert reusable["feasible"] and reusable["selected_source_allocated_bytes"] == positive_credit
        run("plan", farm, "--harvester", reusable_profile, *options, code=2)
        assert snapshot(farm) == guarded, "resource-only profile reuse changed the farm"
    malformed_reader = directory / "malformed combined reader profile.json"
    malformed_profile = json.loads(json.dumps(profile))
    malformed_profile["model"] = "wrong"
    malformed_reader.write_text(json.dumps(malformed_profile), encoding="utf-8")
    run("plan", farm, "--replot", inventory_file, "--harvester", malformed_reader, *adapter, *options, code=2)
    assert snapshot(farm) == guarded, "malformed ordinary reader inputs changed the farm"
    accepted = json.loads(run("plan", farm, "--replot", inventory_file, "--harvester", external_file,
                              *adapter, *external_options, "--reclaim-before-bytes", positive_credit,
                              "--save-plan", external_review, "--json").stdout)
    assert accepted["feasible"] and accepted["reclaim_before_bytes"] == positive_credit and snapshot(farm) == guarded
    run("fill", external_review, "--harvester", external_file, *adapter, *external_options, *keys,
        "--temp-capacity-bytes", 1, code=2)
    assert snapshot(farm) == guarded and not (farm / "grow-replot.plan").exists(), "joint scratch refusal claimed a source or published a journal"
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
    invalid_saved = json.loads(run("plan", farm, "--replot", inventory_file, "--harvester", temporary_file, *adapter, *options,
                                   "--temp-dir", invalid_temp, "--reclaim-before-bytes", 1, "--save-plan", bad_temp_review,
                                   "--json", code=2).stdout)
    assert invalid_saved["infeasibility_reasons"] == invalid["infeasibility_reasons"] and not invalid_saved["saved_plan"]
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
    if temporary_review.read_bytes().startswith(b"ordinary-group-replot-review-v2\n"):
        temporary_before = snapshot(valid_temp), snapshot(external_scratch), digest(temporary_review)
        unreviewed = json.loads(run("plan", temporary_review, "--harvester", temporary_file, *adapter, *options,
                                   "--temp-dir", external_scratch, "--json", code=2).stdout)
        assert any("keep the reviewed scratch scope" in reason for reason in unreviewed["infeasibility_reasons"])
        run("fill", temporary_review, "--harvester", temporary_file, *adapter, *options,
            "--temp-dir", external_scratch, *keys, code=2)
        assert snapshot(farm) == guarded and not (farm / "grow-replot.plan").exists()
        assert (snapshot(valid_temp), snapshot(external_scratch), digest(temporary_review)) == temporary_before
    protected_root = directory / "protected old plots"
    protected_root.mkdir()
    protected_source = protected_root / "protected.plot"
    id_start = len(b"Proof of Space Plot")
    protected_id = hashlib.sha256(bytes.fromhex(POOL + FARMING_PLOT_KEY)).digest()
    protected_source.write_bytes(legacy[:id_start] + protected_id + legacy[id_start + len(plot_id):])
    protected_before = snapshot(protected_root)
    flags(roots=[str(farm), str(protected_root)], legacy_plot_public_keys={str(protected_source): FARMING_PLOT_KEY})
    scoped_inventory_file = directory / "selected and protected.inventory"
    scoped_catalog = json.loads(run("plan", "--inventory", farm, protected_root, *adapter,
                                    "--save-plan", scoped_inventory_file, "--json").stdout)
    scoped_profile = json.loads(json.dumps(profile))
    scoped_counts = {w["id"]: w["plot_count"] for w in scoped_catalog["farm_resources"]["workloads"]}
    for workload in scoped_profile["farm_resources"]["workloads"]:
        workload["plot_count"] = scoped_counts[workload["id"]]
    scoped_profile_file = directory / "selected and protected profile.json"
    scoped_profile_file.write_text(json.dumps(scoped_profile), encoding="utf-8")
    scoped = json.loads(run("plan", farm, "--replot", scoped_inventory_file, "--harvester", scoped_profile_file,
                            *adapter, *options, "--replot-source-root", farm, "--replot-exclude-root", protected_root,
                            "--replot-max-source-files", 1, "--replot-max-source-bytes", positive_credit, "--json").stdout)
    protected_plot = next(p for p in scoped_catalog["plots"] if scoped_catalog["roots"][p["root"]]["path"] == str(protected_root))
    assert scoped["feasible"] and scoped["selected_source_files"] == scoped["protected_inventory_entries"] == 1
    assert scoped["selected_source_allocated_bytes"] == positive_credit
    assert scoped["protected_allocated_bytes"] == protected_plot["identity"]["allocated_bytes"]
    assert snapshot(protected_root) == protected_before and snapshot(farm) == guarded
    protected_source.unlink()
    protected_root.rmdir()
    flags()
    review_file = directory / "migration.plan"
    before = snapshot(farm)
    selection = ["--replot-source-root", farm, "--replot-max-source-files", 1,
                 "--replot-max-source-bytes", positive_credit]
    for selectors in (["--replot-source-root", directory / "unreviewed root"],
                      ["--replot-exclude-root", farm], ["--replot-max-source-bytes", 1]):
        run("plan", farm, "--replot", inventory_file, "--harvester", profile_file,
            *adapter, *options, *selectors, "--json", code=2)
        assert snapshot(farm) == before, "source selection refusal changed the farm"
    review = json.loads(run("plan", farm, "--replot", inventory_file, "--harvester", profile_file,
                            *adapter, *options, *selection, "--save-plan", review_file, "--json").stdout)
    assert review["replot_review"] and review["feasible"] and not review["live_readiness_checked"]
    assert review["reclaim_before_bytes"] == 0 and review["farming_value_preserved"] is None
    assert len(review["sources"]) == len(review["batches"]) == 1 and snapshot(farm) == before
    assert review["inventory_entries"] == len(catalog["plots"]) and review["selected_source_files"] == 1
    assert review["selected_source_allocated_bytes"] == positive_credit
    assert review["protected_inventory_entries"] == review["protected_allocated_bytes"] == 0
    final = Path(review["batches"][0]["path"])
    assert final.suffix == ".plot2" and review["job_id"] in final.name
    assert SEED not in review_file.read_text(encoding="utf-8")
    # Compatible old records may contain trailing whitespace. A successor must
    # retain the exact original checksummed body, rather than canonicalizing it.
    review_body = review_file.read_bytes()[:-65] + b"\n"
    review_file.write_bytes(review_body + hashlib.sha256(review_body).hexdigest().encode() + b"\n")
    saved_review = review_file.read_bytes(), review_file.stat().st_mtime_ns
    execute = ["fill", review_file, "--harvester", profile_file, *adapter, *options]
    public_only = run("fill", "--from-plan", review_file, *options, *keys, code=2)
    assert "explicit installed target reader" in public_only.stdout + public_only.stderr and snapshot(farm) == before
    for failure in ("farmer_unavailable", "solver_unavailable", "farmer_disconnected", "solver_stopped", "pool_external", "pool_mismatch", "node_mismatch"):
        flags(**{failure: True})
        unavailable = run(*execute, *keys, code=2)
        assert snapshot(farm) == before and not (farm / "grow-replot.plan").exists()
        if failure == "pool_external":
            assert "pool" in (unavailable.stdout + unavailable.stderr).lower()
    flags()
    for name in ("farmer_version", "solver_version"):
        flags(**{name: "changed-service-version"})
        drift = run(*execute, *keys, code=2)
        assert "service version changed since review" in drift.stdout + drift.stderr and snapshot(farm) == before
    flags()
    run(*execute, *keys, point="replot-journal-all-staged", code=86)
    staged_private = snapshot(farm)
    run(*execute, "-f", FARMER, "--pool-pk", FARMER, "-S", SEED, code=2)
    assert snapshot(farm) == staged_private, "different supplied pool route changed a prepared NFT journal"
    for failure in ("farmer_disconnected", "solver_stopped"):
        flags(**{failure: True})
        run(*execute, code=2)
        assert snapshot(farm) == staged_private, "keyless service refusal changed an interrupted private journal or source"
    flags()
    run(*execute, *keys, point="replot-journal-one-published", code=86)
    assert digest(source) == source_hash and not (farm / "grow-farm.plan").exists()
    saved_work = farm / f".xchplot2-stage-{review['job_id']}-0.building.external"
    saved_work.mkdir(mode=0o700)
    partial_before = snapshot(farm)
    partial_review = run("plan", "--revalidate", "--from-plan", review_file, "--harvester", profile_file,
                         *adapter, *options, "--save-plan", directory / "must-not-review-partial.plan", code=2)
    assert "retained construction lacks its exact stage target" in partial_review.stdout + partial_review.stderr
    assert snapshot(farm) == partial_before and saved_work.is_dir()
    assert digest(source) == source_hash and not (directory / "must-not-review-partial.plan").exists()
    saved_work.rmdir()
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
    local_bindings, = farm.glob(".xchplot2-replot-*/local-bindings")
    assert local_bindings.stat().st_mode & 0o777 == 0o600
    before_status = snapshot(farm)
    flags(farmer_unavailable=True, solver_unavailable=True)
    status = json.loads(run("fill", "--status", "--from-plan", review_file, "--json").stdout)
    assert status["durable_phase"] == "replacing" and status["active_verified_stage"]
    assert status["live_service_health"] == "not-queried" and status["blocking_reason"] is None
    verified_authority = saved_replacement_authority(farm / "grow-replot.plan", "verified_stage")
    assert status["active_replacement"]["current_native_observation"] == f"matching-verified-stage-native-{verified_authority}"
    assert snapshot(farm) == before_status, "status changed saved progress or native objects"
    flags()
    run("fill", "--from-plan", review_file, "--chia-executable", directory / "missing-reader", *options, code=2)
    assert snapshot(farm) == before_status, "explicit local override was ignored or changed progress on refusal"
    saved_local = local_bindings.read_bytes()
    local_bindings.chmod(0o644)
    private_before = snapshot(farm)
    private_refusal = run("fill", "--from-plan", review_file, *options, code=2)
    assert "owner-only permissions" in private_refusal.stdout + private_refusal.stderr and snapshot(farm) == private_before
    local_bindings.chmod(0o600)
    local_rows = saved_local[:-65].decode().splitlines(keepends=True)
    local_identity = shlex.split(local_rows[1])
    local_identity[2] = "foreign-host-fixture"
    local_rows[1] = " ".join(json.dumps(value) for value in local_identity) + "\n"
    foreign_body = "".join(local_rows).encode()
    local_bindings.write_bytes(foreign_body + hashlib.sha256(foreign_body).hexdigest().encode() + b"\n")
    foreign_before = snapshot(farm)
    foreign_refusal = run("fill", "--from-plan", review_file, *options, code=2)
    assert "another host or user" in foreign_refusal.stdout + foreign_refusal.stderr and snapshot(farm) == foreign_before
    run("plan", "--revalidate", "--from-plan", review_file, "--harvester", profile_file, "--farm-adapter", "chia",
        *adapter, *options, "--save-plan", directory / "explicit foreign-default replacement.plan")
    assert snapshot(farm) == foreign_before, "explicit local binding review trusted or changed foreign defaults"
    local_bindings.write_bytes(saved_local)
    before_status = snapshot(farm)
    stale_successor = directory / "stale successor.plan"
    stale = json.loads(run("plan", "--revalidate", "--from-plan", review_file, "--harvester", profile_file,
                           *adapter, *options, "--save-plan", stale_successor, "--json").stdout)
    assert not stale["activated"] and snapshot(farm) == before_status
    assert SEED not in stale_successor.read_text(encoding="utf-8")
    successor_metadata = json.loads(run("plan", "--from-plan", stale_successor, "--json").stdout)
    assert successor_metadata["saved_successor"] and successor_metadata["metadata_only"] and successor_metadata["feasible"] is None
    assert successor_metadata["selected_policy_digest"] == stale["successor_digest"]
    assert successor_metadata["reviewed_journal_digest"] == stale["reviewed_journal_digest"] and snapshot(farm) == before_status
    flags(legacy_plot_public_keys={str(source): "ab" * 48})
    imported_mismatch = run("fill", "--from-plan", stale_successor, "--harvester", profile_file, *adapter, *options, code=2)
    assert "original public key" in imported_mismatch.stdout + imported_mismatch.stderr
    assert snapshot(farm) == before_status, "unverified imported source metadata activated a policy or changed saved local defaults"
    flags()
    # Saved private defaults restore the accepted local profile and reader;
    # no keys, --harvester or installed-reader flags are repeated.
    execute = ["fill", "--from-plan", review_file, *options]
    staged, = farm.glob(".xchplot2-stage-*.building")
    guarded = snapshot(farm)
    direct = run("grow", staged, "--max-members", 2, "--devices", "cpu", "--compute", "off", "-M", ram, code=2)
    assert "migration-managed directory" in direct.stdout + direct.stderr
    assert snapshot(farm) == guarded, "direct growth changed an active migration's sealed stage"
    run(*execute, point="replot-replacement-published", code=86)
    replacement_hash = digest(final)
    before_status = snapshot(farm)
    renamed_status = json.loads(run("plan", "--status", "--from-plan", review_file, "--json").stdout)
    assert renamed_status["journal_digest"] == stale["reviewed_journal_digest"] and not renamed_status["active_published"]
    assert renamed_status["active_replacement"]["current_native_observation"] == "verified-object-renamed-publication-pending"
    assert snapshot(farm) == before_status, "renamed replacement status changed saved progress or native objects"
    # Native publication precedes the durable published stamp. Preserve that
    # supported rename recovery, then advance the journal before testing stale
    # successor refusal; readiness has not yet been acknowledged.
    run(*execute, point="replot-journal-one-published", code=86)
    before_status = snapshot(farm)
    published_status = json.loads(run("plan", "--status", "--from-plan", review_file, "--json").stdout)
    assert published_status["active_published"] and not published_status["active_ready_history"]
    published_authority = saved_replacement_authority(farm / "grow-replot.plan", "published")
    assert published_status["active_replacement"]["current_native_observation"] == f"matching-published-native-{published_authority}"
    assert published_status["journal_digest"] != stale["reviewed_journal_digest"] and digest(final) == replacement_hash
    assert snapshot(farm) == before_status, "published replacement status changed saved progress or native objects"
    guarded = snapshot(farm)
    stale_attempt = run("fill", "--from-plan", stale_successor, *options, code=2)
    assert "advanced after this successor was reviewed" in stale_attempt.stdout + stale_attempt.stderr
    assert snapshot(farm) == guarded, "stale unactivated successor changed progress"
    stale_status = json.loads(run("plan", "--status", "--from-plan", stale_successor, "--json").stdout)
    assert stale_status["selected_policy_binding"] == "stale-boundary"
    assert stale_status["reviewed_journal_digest"] == stale["reviewed_journal_digest"] != stale_status["journal_digest"]
    assert snapshot(farm) == guarded, "stale successor status changed progress"
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
    claimed_status = json.loads(run("plan", "--status", "--from-plan", review_file, "--json").stdout)
    assert claimed_status["active_sources"][0]["current_native_observation"] == "exact-claimed-object-present"
    updated_profile = json.loads(profile_file.read_text(encoding="utf-8"))
    for resource in updated_profile["farm_resources"]["resources"]:
        if resource["kind"] == "host_memory":
            resource["capacity"] += ram
    updated_file = directory / "updated local farm.json"
    updated_file.write_text(json.dumps(updated_profile), encoding="utf-8")
    (reader / "config" / "config.yaml").write_text("self_hostname: localhost\nharvester:\n  num_threads: 2\n", encoding="utf-8")
    guarded = snapshot(farm)
    run(*execute, code=2)
    assert snapshot(farm) == guarded, "unreviewed reader drift changed a claimed source"
    successor_file = directory / "active successor.plan"
    successor = json.loads(run("plan", "--revalidate", "--from-plan", review_file, "--harvester", updated_file,
                               *adapter, *options, "--farm-adapter-timeout-ms", 120000, "--save-plan", successor_file, "--json").stdout)
    assert not successor["activated"] and snapshot(farm) == guarded and SEED not in successor_file.read_text(encoding="utf-8")
    execute = ["fill", "--from-plan", successor_file, "--harvester", updated_file, *adapter, *options]
    run(*execute, point="replot-policy-durable", code=86)
    activated = snapshot(farm)
    run("fill", "--from-plan", review_file, *options, code=2)
    assert snapshot(farm) == activated, "original review dropped an active successor policy"
    active_status = json.loads(run("plan", "--status", "--from-plan", successor_file, "--json").stdout)
    assert active_status["active_policy_digest"] == successor["successor_digest"]
    assert active_status["origin_review_digest"] == successor["origin_review_digest"]
    assert active_status["selected_policy_binding"] == "active"
    origin_status = json.loads(run("plan", "--status", "--from-plan", review_file, "--json").stdout)
    assert origin_status["selected_policy_binding"] == "origin-with-active-successor"
    # The real activation-before-defaults crash retains the old local file.
    # A single route can bind the successor's embedded policy without reusing
    # that stale profile; then an explicit accepted profile is saved again.
    run("fill", "--from-plan", successor_file, *options, "--farm-adapter-timeout-ms", 120000, point="replot-before-unlink", code=86)
    run(*execute, point="replot-before-unlink", code=86)
    execute = ["fill", "--from-plan", successor_file, *options]
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

    # Review another policy after an immutable completed prefix. Both policy
    # and ordinary progress revisions must recover across destination replicas.
    chain_old = directory / "successor chain old"
    chain_destinations = [directory / "successor chain left", directory / "successor chain right"]
    for path in [chain_old, *chain_destinations]:
        path.mkdir()
    chain_sources = [chain_old / "first.plot", chain_old / "second.plot"]
    chain_sources[0].write_bytes(legacy)
    chain_sources[1].write_bytes(legacy[:id_start] + protected_id + legacy[id_start + len(plot_id):])
    chain_reader = directory / "successor chain reader"
    (chain_reader / "config").mkdir(parents=True)
    chain_config = chain_reader / "config" / "config.yaml"

    def chain_threads(threads):
        chain_config.write_text(f"self_hostname: localhost\nharvester:\n  num_threads: {threads}\n", encoding="utf-8")

    chain_threads(1)
    (chain_reader / "fixture.json").write_text(json.dumps({
        "roots": list(map(str, [chain_old, *chain_destinations])),
        "legacy_plot_public_keys": {str(chain_sources[1]): FARMING_PLOT_KEY}}), encoding="utf-8")
    chain_adapter = ["--chia-root", chain_reader, "--chia-executable", executable]
    chain_inventory = directory / "successor chain inventory.plan"
    chain_catalog = json.loads(run("plan", "--inventory", chain_old, *chain_adapter,
                                   "--save-plan", chain_inventory, "--json").stdout)
    chain_profile = json.loads(json.dumps(profile))
    for workload in chain_profile["farm_resources"]["workloads"]:
        workload["plot_count"] = len(chain_catalog["plots"])
    chain_profile_file = directory / "successor chain profile.json"
    chain_profile_file.write_text(json.dumps(chain_profile), encoding="utf-8")
    chain_options = list(options)
    chain_review_file = directory / "successor chain origin.plan"
    chain_review = json.loads(run("plan", *chain_destinations, "--replot", chain_inventory, "--harvester", chain_profile_file,
                                  *chain_adapter, *chain_options, "--save-plan", chain_review_file, "--json").stdout)
    assert [batch["source_count"] for batch in chain_review["batches"]] == [2, 0]
    assert {Path(batch["path"]).parent for batch in chain_review["batches"]} == set(chain_destinations)
    replicas = [Path(item["directory"]) / "grow-replot.plan" for item in chain_review["space"]]
    run("fill", "--from-plan", chain_review_file, "--harvester", chain_profile_file, *chain_adapter, *chain_options, *keys,
        point=f"replot-journal-one-published:{len(replicas)}", code=86)

    def private_replot_farm(path):
        return shlex.split(path.read_bytes()[:-65].decode("utf-8"))[8].encode("utf-8")

    def chain_private_farms():
        return [private_replot_farm(path) for path in replicas]

    private_farm = chain_private_farms()[0]
    assert all(body == private_farm for body in chain_private_farms()), "preparation replicas disagree on private identity"
    chain_threads(2)
    chain_first_profile = directory / "successor chain first profile.json"
    chain_first_profile.write_text(json.dumps(chain_profile), encoding="utf-8")
    chain_first_file = directory / "successor chain first.plan"
    before = tuple(snapshot(path) for path in [chain_old, *chain_destinations])
    chain_first = json.loads(run("plan", "--revalidate", "--from-plan", chain_review_file, "--harvester", chain_first_profile,
                                 *chain_adapter, *chain_options, "--save-plan", chain_first_file, "--json").stdout)
    assert tuple(snapshot(path) for path in [chain_old, *chain_destinations]) == before
    pending = json.loads(run("plan", "--status", "--from-plan", chain_first_file, "--json").stdout)
    assert pending["selected_policy_binding"] == "pending" and pending["reviewed_journal_digest"] == pending["journal_digest"]
    run("fill", "--from-plan", chain_first_file, "--harvester", chain_first_profile, *chain_adapter, *chain_options,
        point="replot-policy-activated", code=86)
    run("fill", "--from-plan", chain_first_file, *chain_options, point="replot-receipt-synced", code=86)
    chain_outputs = [Path(batch["path"]) for batch in chain_review["batches"]]
    receipt = chain_outputs[0].parent / f".xchplot2-replot-{chain_review['job_id']}" / "receipt-0"
    receipt_bytes, prefix_hash = receipt.read_bytes(), digest(chain_outputs[0])
    # Receipt publication precedes prefix advancement. Here the next journal
    # update is exactly complete_replot_batch; stop after every replica commits.
    run("fill", "--from-plan", chain_first_file, *chain_options,
        point=f"replot-journal-one-published:{len(replicas)}", code=86)
    prefix = json.loads(run("plan", "--status", "--from-plan", chain_first_file, "--json").stdout)
    assert prefix["completed_replacements"] == 1 and prefix["completed_retirements"] == len(chain_sources)
    assert prefix["completed_receipt_chain_verified"] and not prefix["replica_publication_pending"]
    completed_output, = prefix["completed_outputs"]
    assert completed_output["current_native_observation"] == f"matching-saved-native-{saved_replacement_authority(receipt)}"
    assert not list(chain_outputs[1].parent.glob(".xchplot2-stage-*.building"))
    assert receipt.read_bytes() == receipt_bytes and digest(chain_outputs[0]) == prefix_hash
    assert all(body == private_farm for body in chain_private_farms()), "completed prefix changed private farm identity"

    chain_threads(3)
    chain_second_profile = directory / "successor chain second profile.json"
    chain_second_profile.write_text(json.dumps(chain_profile), encoding="utf-8")
    chain_second_file = directory / "successor chain second.plan"
    before = tuple(snapshot(path) for path in [chain_old, *chain_destinations])
    chain_second = json.loads(run("plan", "--revalidate", "--from-plan", chain_first_file, "--harvester", chain_second_profile,
                                  *chain_adapter, *chain_options, "--save-plan", chain_second_file, "--json").stdout)
    assert chain_second["parent_policy_digest"] == chain_first["successor_digest"]
    assert chain_second["origin_review_digest"] == chain_first["origin_review_digest"] and chain_second["completed_replacements"] == 1
    successor_text = chain_second_file.read_text(encoding="utf-8")
    assert sum(successor_text.count("ordinary-group-replot-review-v" + version) for version in ("1", "2")) == 1
    embedded_origin = shlex.split(farming_plan_rows(chain_second_file.read_bytes()[:-65])[1].decode("utf-8"))[0].encode("utf-8")
    assert embedded_origin == chain_review_file.read_bytes()[:-65]
    assert hashlib.sha256(embedded_origin).hexdigest() == chain_second["origin_review_digest"]
    assert tuple(snapshot(path) for path in [chain_old, *chain_destinations]) == before
    run("fill", "--from-plan", chain_second_file, "--harvester", chain_second_profile, *chain_adapter, *chain_options,
        point="replot-journal-one-published", code=86)
    partial = json.loads(run("plan", "--status", "--from-plan", chain_second_file, "--json").stdout)
    assert partial["selected_policy_binding"] == "active" and partial["replica_publication_pending"]
    assert len({replica["published_digest"] for replica in partial["replicas"]}) > 1
    assert partial["completed_replacements"] == 1 and receipt.read_bytes() == receipt_bytes
    before = tuple(snapshot(path) for path in [chain_old, *chain_destinations])
    for prior in (chain_review_file, chain_first_file):
        run("fill", "--from-plan", prior, *chain_options, code=2)
        assert tuple(snapshot(path) for path in [chain_old, *chain_destinations]) == before, "prior policy changed a partially published successor"
    prior_status = json.loads(run("plan", "--status", "--from-plan", chain_first_file, "--json").stdout)
    assert prior_status["selected_policy_binding"] == "different-active-policy"
    run("fill", "--from-plan", chain_second_file, "--harvester", chain_second_profile, *chain_adapter, *chain_options,
        point="replot-journal-one-published", code=86)
    reconciled = json.loads(run("plan", "--status", "--from-plan", chain_second_file, "--json").stdout)
    assert not reconciled["replica_publication_pending"] and reconciled["active_policy_digest"] == chain_second["successor_digest"]
    assert receipt.read_bytes() == receipt_bytes and digest(chain_outputs[0]) == prefix_hash
    assert all(body == private_farm for body in chain_private_farms()), "successor policy changed saved keys or fixed assignments"
    run("fill", "--from-plan", chain_second_file, "--harvester", chain_second_profile, *chain_adapter, *chain_options)
    assert not any(path.exists() for path in chain_sources)
    assert receipt.read_bytes() == receipt_bytes and digest(chain_outputs[0]) == prefix_hash
    assert all(body == private_farm for body in chain_private_farms())
    for output in chain_outputs:
        proof = run("verify", output, "--full", "--trials", 100)
        found = re.search(r"(\d+) full proofs validated", (proof.stdout + proof.stderr).lower())
        assert found and int(found[1]) > 0
    print("PASS repeated reviewed successors preserve completed receipts and private identity across interrupted replica publication", flush=True)

    # Journal progress alone does not bind an unfinished engine checkpoint.
    for engine, initial_point, advance_point, active_point in (
            ("incremental", "marked", "direct-committed", "finalizing"),
            ("staged", "stage-run-committed", "stage-pack-committed", "stage-tail-committed"),
            ("merged", "merge-run-committed", "merge-pack-committed", "merge-layout-ready-committed"),
            ("external", "external-run-after-checkpoint", "external-pack-after-checkpoint", "external-ready-after-checkpoint")):
        partial_farm = directory / ("partial successor " + engine)
        partial_farm.mkdir()
        partial_source = partial_farm / "legacy.plot"
        partial_source.write_bytes(legacy)
        partial_reader = directory / ("partial reader " + engine)
        (partial_reader / "config").mkdir(parents=True)
        partial_config = partial_reader / "config" / "config.yaml"

        def partial_threads(threads):
            partial_config.write_text(f"self_hostname: localhost\nharvester:\n  num_threads: {threads}\n", encoding="utf-8")

        partial_threads(1)
        (partial_reader / "fixture.json").write_text(json.dumps({"roots": [str(partial_farm)]}), encoding="utf-8")
        partial_adapter = ["--chia-root", partial_reader, "--chia-executable", executable]
        partial_inventory = directory / (engine + " partial inventory.plan")
        run("plan", "--inventory", partial_farm, *partial_adapter, "--save-plan", partial_inventory, "--json")
        partial_profile = json.loads(json.dumps(profile))
        partial_profile["farm_resources"]["native"].update(engine=engine, scratch_resources=disks if engine == "external" else [])
        partial_profile_file = directory / (engine + " partial profile.json")
        partial_profile_file.write_text(json.dumps(partial_profile), encoding="utf-8")
        partial_options = list(external_options if engine == "external" else options)
        partial_options[partial_options.index("--construction") + 1] = engine
        partial_origin = directory / (engine + " partial origin.plan")
        partial_review = json.loads(run("plan", partial_farm, "--replot", partial_inventory, "--harvester", partial_profile_file,
                                        *partial_adapter, *partial_options, "--save-plan", partial_origin, "--json").stdout)
        run("fill", "--from-plan", partial_origin, "--harvester", partial_profile_file, *partial_adapter, *partial_options, *keys,
            point=initial_point, code=86)
        initial_status = json.loads(run("plan", "--status", "--from-plan", partial_origin, "--json").stdout)
        assert not initial_status["active_verified_stage"] and not initial_status["active_published"]
        private_path = partial_farm / "grow-replot.plan"
        partial_private_farm = private_replot_farm(private_path)
        partial_threads(2)
        partial_successor = directory / (engine + " partial successor.plan")
        before = snapshot(partial_farm)
        run("plan", "--revalidate", "--from-plan", partial_origin, "--harvester", partial_profile_file,
            *partial_adapter, *partial_options, "--save-plan", partial_successor, "--json")
        assert snapshot(partial_farm) == before and partial_source.read_bytes() == legacy
        assert partial_successor.read_bytes().startswith(b"ordinary-group-replot-revalidation-v2\n")
        evidence = shlex.split(farming_plan_rows(partial_successor.read_bytes()[:-65])[-1].decode("utf-8"))
        assert evidence[1] == engine and len(evidence[2]) == len(evidence[3]) == 64
        wrong_engine = json.loads(json.dumps(partial_profile))
        wrong_engine["farm_resources"]["native"]["engine"] = "staged" if engine != "staged" else "incremental"
        wrong_profile = directory / (engine + " wrong engine.json")
        wrong_profile.write_text(json.dumps(wrong_engine), encoding="utf-8")
        rejected_review = directory / (engine + " rejected successor.plan")
        run("plan", "--revalidate", "--from-plan", partial_origin, "--harvester", wrong_profile,
            *partial_adapter, *partial_options, "--save-plan", rejected_review, code=2)
        assert not rejected_review.exists() and snapshot(partial_farm) == before

        # Resume the origin only to another durable engine checkpoint. The
        # reviewed private journal stays unchanged, so its digest is insufficient.
        partial_threads(1)
        run("fill", "--from-plan", partial_origin, "--harvester", partial_profile_file, *partial_adapter, *partial_options,
            point=advance_point, code=86)
        advanced = json.loads(run("plan", "--status", "--from-plan", partial_origin, "--json").stdout)
        assert advanced["journal_digest"] == initial_status["journal_digest"] and not advanced["active_verified_stage"]
        assert private_replot_farm(private_path) == partial_private_farm
        partial_threads(2)
        before = snapshot(partial_farm)
        stale = run("fill", "--from-plan", partial_successor, "--harvester", partial_profile_file,
                    *partial_adapter, *partial_options, code=2)
        assert "checkpoint or native stage changed" in stale.stdout + stale.stderr and snapshot(partial_farm) == before
        current_successor = directory / (engine + " current partial successor.plan")
        current = json.loads(run("plan", "--revalidate", "--from-plan", partial_origin, "--harvester", partial_profile_file,
                                  *partial_adapter, *partial_options, "--save-plan", current_successor, "--json").stdout)
        assert snapshot(partial_farm) == before
        run("fill", "--from-plan", current_successor, "--harvester", partial_profile_file, *partial_adapter, *partial_options,
            point="replot-policy-activated", code=86)
        activated = json.loads(run("plan", "--status", "--from-plan", current_successor, "--json").stdout)
        engine_before = snapshot(partial_farm)
        run("fill", "--from-plan", current_successor, *partial_options, point=active_point, code=86)
        progressing = json.loads(run("plan", "--status", "--from-plan", current_successor, "--json").stdout)
        assert progressing["journal_digest"] == activated["journal_digest"] and snapshot(partial_farm) != engine_before
        assert progressing["active_policy_digest"] == current["successor_digest"] and private_replot_farm(private_path) == partial_private_farm
        run("fill", "--from-plan", current_successor, *partial_options)
        assert not partial_source.exists() and private_replot_farm(private_path) == partial_private_farm
        output = Path(partial_review["batches"][0]["path"])
        proof = run("verify", output, "--full", "--trials", 100)
        found = re.search(r"(\d+) full proofs validated", (proof.stdout + proof.stderr).lower())
        assert found and int(found[1]) > 0
    print("PASS unfinished exact-build checkpoints bind successors and continue after reviewed activation", flush=True)

    # Source and target implementations use distinct explicit local routes.
    # The old route deliberately has no compatible target full-node interface.
    routed_farm = directory / "routed farm"
    routed_farm.mkdir()
    routed_source = routed_farm / "legacy.plot"
    routed_source.write_bytes(legacy)
    route_specs = []
    for name, kind, extra in (("old-gh", "gigahorse", {"source_only": True, "version": "fixture-gh-v1"}),
                              ("new-chia", "chia", {"target_only": True, "pool_external": True, "pool_protocol_version": 2})):
        config = directory / name
        (config / "config").mkdir(parents=True)
        (config / "config" / "config.yaml").write_text("self_hostname: localhost\nharvester:\n  num_threads: 1\n", encoding="utf-8")
        (config / "fixture.json").write_text(json.dumps({"roots": [str(routed_farm)], **extra}), encoding="utf-8")
        route_specs.append({"id": name, "kind": kind, "root": str(config), "executable": str(executable),
                            "source_roots": [str(routed_farm)] if name == "old-gh" else []})
    routed_profile = json.loads(json.dumps(profile))
    routed_profile["farm_adapters"] = {"schema": 1, "target_route": "new-chia", "routes": route_specs}
    routed_profile["farm_resources"]["native"].update(min_pool_difficulty=17, pool_protocol_version=2, engine="external", scratch_resources=disks)
    routed_file = directory / "routed profile.json"
    routed_file.write_text(json.dumps(routed_profile), encoding="utf-8")
    routed_inventory = directory / "routed inventory.plan"
    observed = json.loads(run("plan", "--inventory", routed_farm, "--harvester", routed_file,
                              "--save-plan", routed_inventory, "--json").stdout)
    assert observed["plots"][0]["harvester"] == "old-gh" and observed["network"]["target_route"] == "new-chia"
    routed_review = directory / "routed migration.plan"
    reviewed = json.loads(run("plan", routed_farm, "--replot", routed_inventory, "--harvester", routed_file,
                              *external_options, "--reclaim-before-bytes", observed["plots"][0]["identity"]["allocated_bytes"], "--save-plan", routed_review, "--json").stdout)
    assert reviewed["feasible"] and reviewed["reclaim_before_bytes"] > 0
    before = snapshot(routed_farm)
    run("fill", routed_review, *keys, "--harvester", routed_file, *external_options, "--temp-capacity-bytes", 1, code=2)
    assert snapshot(routed_farm) == before and not (routed_farm / "grow-replot.plan").exists()
    target_fixture = Path(route_specs[1]["root"]) / "fixture.json"
    target_original = target_fixture.read_bytes()
    for change in ({"pool_difficulty": 16}, {"pool_protocol_version": 1}, {"pool_uninitialized": True}):
        target_fixture.write_text(json.dumps({**json.loads(target_original), **change}), encoding="utf-8")
        rejected_pool = run("fill", routed_review, *keys, "--harvester", routed_file, *external_options, code=2)
        failure = (rejected_pool.stdout + rejected_pool.stderr).lower()
        assert ("pool" in failure or (change.get("pool_uninitialized") and "integer" in failure)) and snapshot(routed_farm) == before
    target_fixture.write_bytes(target_original)
    run("fill", routed_review, *keys, *adapter, *external_options, code=2)
    assert snapshot(routed_farm) == before, "saved public route IDs authorized a missing local adapter profile"
    swapped = json.loads(json.dumps(routed_profile))
    swapped["farm_adapters"]["target_route"] = "old-gh"
    wrong_route = directory / "wrong route.json"
    wrong_route.write_text(json.dumps(swapped), encoding="utf-8")
    run("fill", routed_review, *keys, "--harvester", wrong_route, *external_options, code=2)
    assert snapshot(routed_farm) == before, "changed target route wrote before rejecting the saved binding"
    uncovered = json.loads(json.dumps(routed_profile))
    uncovered["farm_adapters"]["routes"][0]["source_roots"] = []
    wrong_route.write_text(json.dumps(uncovered), encoding="utf-8")
    run("fill", routed_review, *keys, "--harvester", wrong_route, *external_options, code=2)
    assert snapshot(routed_farm) == before, "missing source route wrote before rejection"
    run("fill", routed_review, *keys, "--harvester", routed_file, *external_options,
        point="replot-replacement-published", code=86)
    routed_final = Path(reviewed["batches"][0]["path"])
    routed_hash = digest(routed_final)
    old_fixture = Path(route_specs[0]["root"]) / "fixture.json"
    original_fixture = old_fixture.read_bytes()
    try:
        old_fixture.write_text(json.dumps({"roots": [str(routed_farm)], "version": "fixture-gh-v1"}), encoding="utf-8")
        doubled = run("fill", routed_review, "--harvester", routed_file, *external_options, code=2)
        assert "outside the reviewed reader/catalog scope" in doubled.stdout + doubled.stderr
        assert routed_source.exists() and digest(routed_final) == routed_hash
    finally:
        old_fixture.write_bytes(original_fixture)
    run("fill", routed_review, "--harvester", routed_file, *external_options)
    routed_final = Path(reviewed["batches"][0]["path"])
    assert not routed_source.exists() and digest(routed_final) == replacement_hash
    assert (routed_farm / "grow-farm.plan").is_file()
    print("PASS distinct source/target adapters, initialized RewardPuzzle v2 pooling, external positive credit and reviewed difficulty/protocol gates", flush=True)

    # Mutate only the old reader while this invocation constructs its real
    # replacement, or after it claims the old object. A restart-only preflight
    # cannot detect the first event and target-only leases cannot detect either.
    for when in ("published", "claimed"):
        for change in (("config", "version", "population") if when == "published" else
                       ("config", "version", "population", "target_population")):
            guarded_farm = directory / f"source guard {when} {change}"
            guarded_farm.mkdir()
            guarded_source = guarded_farm / "legacy.plot"
            guarded_source.write_bytes(legacy)
            guarded_hash = digest(guarded_source)
            guarded_routes = []
            for spec in route_specs:
                config = directory / f"{when} {change} {spec['id']}"
                (config / "config").mkdir(parents=True)
                (config / "config" / "config.yaml").write_text("self_hostname: localhost\nharvester:\n  num_threads: 1\n", encoding="utf-8")
                is_source = spec["id"] == "old-gh"
                settings = {"roots": [str(guarded_farm)], "source_only" if is_source else "target_only": True}
                if is_source:
                    settings["version"] = "fixture-gh-v1"
                (config / "fixture.json").write_text(json.dumps(settings), encoding="utf-8")
                guarded_routes.append({**spec, "root": str(config), "source_roots": [str(guarded_farm)] if is_source else []})
            guarded_profile = json.loads(json.dumps(profile))
            guarded_profile["farm_adapters"] = {"schema": 1, "target_route": "new-chia", "routes": guarded_routes}
            guarded_file = directory / f"source guard {when} {change}.json"
            guarded_file.write_text(json.dumps(guarded_profile), encoding="utf-8")
            guarded_inventory = directory / f"source guard {when} {change}.inventory"
            run("plan", "--inventory", guarded_farm, "--harvester", guarded_file, "--save-plan", guarded_inventory, "--json")
            guarded_review = directory / f"source guard {when} {change}.plan"
            planned = json.loads(run("plan", guarded_farm, "--replot", guarded_inventory, "--harvester", guarded_file,
                                     *options, "--save-plan", guarded_review, "--json").stdout)
            assert planned["feasible"]
            target = Path(guarded_routes[1]["root"])
            settings = json.loads((target / "fixture.json").read_text(encoding="utf-8"))
            settings["mutate_source"] = {"when": when, "change": change, "root": guarded_routes[0]["root"], "source": str(guarded_source)}
            (target / "fixture.json").write_text(json.dumps(settings), encoding="utf-8")
            refused = run("fill", guarded_review, "--harvester", guarded_file, *options, *keys, code=2)
            assert (target / "source-mutated").read_text(encoding="utf-8") == change
            failure = (refused.stdout + refused.stderr).lower()
            assert "reader" in failure or "scope" in failure, failure
            claims = list(guarded_farm.glob(".xchplot2-retire-*/source-*"))
            if when == "published":
                assert guarded_source.exists() and digest(guarded_source) == guarded_hash and not claims
            else:
                assert not guarded_source.exists() and len(claims) == 1 and digest(claims[0]) == guarded_hash
            assert not (guarded_farm / "grow-farm.plan").exists()
    print("PASS in-process source configuration/version/population fences before claim and unlink", flush=True)

    # A native plots-remove operation rewrites YAML as well as unloading the
    # source directory. Preserve the replacement directory and every other
    # setting, for distinct readers and one shared source/target Chia reader.
    for shared in (False, True):
        name = "shared" if shared else "distinct"
        old = directory / f"{name} unload old"
        destination = directory / f"{name} unload target"
        old.mkdir()
        destination.mkdir()
        old_source = old / "legacy.plot"
        old_source.write_bytes(legacy)
        old_hash = digest(old_source)
        old_reader = directory / f"{name} unload reader"
        new_reader = old_reader if shared else directory / f"{name} unload target reader"
        for config in {old_reader, new_reader}:
            (config / "config").mkdir(parents=True)

        def config_text(roots, rewritten=False):
            dirs = ("  plot_directories:\n" + "".join("  - " + json.dumps(str(path)) + "\n" for path in roots)) if roots else "  plot_directories: []\n"
            if rewritten:
                return "harvester:\n  num_threads: 1\n" + dirs + "self_hostname: localhost\n"
            return "# initial conventional config\nself_hostname: &self_hostname \"localhost\"\nharvester:\n  num_threads: 1\n" + dirs

        original_config = config_text([old, destination] if shared else [old])
        (old_reader / "config" / "config.yaml").write_text(original_config, encoding="utf-8")
        old_settings = {"roots": [str(old), str(destination)] if shared else [str(old)]}
        if not shared:
            old_settings.update(source_only=True, version="fixture-gh-v1")
            (new_reader / "config" / "config.yaml").write_text(config_text([destination]), encoding="utf-8")
            (new_reader / "fixture.json").write_text(json.dumps({"roots": [str(destination)], "target_only": True}), encoding="utf-8")
        (old_reader / "fixture.json").write_text(json.dumps(old_settings), encoding="utf-8")
        unload_routes = [{"id": "new-chia" if shared else "old-gh", "kind": "chia" if shared else "gigahorse",
                          "root": str(old_reader), "executable": str(executable),
                          "source_roots": [str(old), str(destination)] if shared else [str(old)]}]
        if not shared:
            unload_routes.append({"id": "new-chia", "kind": "chia", "root": str(new_reader), "executable": str(executable),
                                  "source_roots": [str(destination)]})
        unload_profile = json.loads(json.dumps(profile))
        unload_profile["farm_adapters"] = {"schema": 1, "target_route": "new-chia", "routes": unload_routes}
        unload_file = directory / f"{name} unload profile.json"
        unload_file.write_text(json.dumps(unload_profile), encoding="utf-8")
        unload_inventory = directory / f"{name} unload inventory.plan"
        unload_catalog = json.loads(run("plan", "--inventory", old, destination, "--harvester", unload_file,
                                        "--save-plan", unload_inventory, "--json").stdout)
        assert unload_catalog["retirement_readers"]
        unload_review = directory / f"{name} unload migration.plan"
        planned = json.loads(run("plan", destination, "--replot", unload_inventory, "--harvester", unload_file,
                                 *options, "--save-plan", unload_review, "--json").stdout)
        assert planned["feasible"]
        after = config_text([destination] if shared else [], rewritten=True)
        # Public review alone cannot authorize a changed reader context.
        (old_reader / "config" / "config.yaml").write_text(after, encoding="utf-8")
        run("fill", unload_review, "--harvester", unload_file, *options, *keys, code=2)
        assert old_source.exists() and digest(old_source) == old_hash and not (destination / "grow-replot.plan").exists()
        (old_reader / "config" / "config.yaml").write_text(original_config, encoding="utf-8")
        run("fill", unload_review, "--harvester", unload_file, *options, *keys, point="replot-retirement-intent", code=86)
        assert old_source.exists() and digest(old_source) == old_hash
        (old_reader / "config" / "config.yaml").write_text(after, encoding="utf-8")
        (old_reader / "fixture.json").write_text(json.dumps({**old_settings, "roots": [str(destination)] if shared else []}), encoding="utf-8")
        run("fill", unload_review, "--harvester", unload_file, *options)
        final_unload = Path(planned["batches"][0]["path"])
        assert not old_source.exists() and digest(final_unload) == replacement_hash and (destination / "grow-farm.plan").is_file()
    print("PASS durable reviewed source-directory removal with distinct and shared target configurations", flush=True)

    # Resource-only bootstrap: actual disk space is ample. Complete serialized
    # CPU envelopes fit separately; their simultaneous response does not fit.
    resource_farm = directory / "resource-only migration"
    resource_farm.mkdir()
    resource_source = resource_farm / "legacy.plot"
    resource_source.write_bytes(legacy)
    resource_reader = directory / "resource-only reader"
    (resource_reader / "config").mkdir(parents=True)
    (resource_reader / "config" / "config.yaml").write_text("self_hostname: localhost\nharvester:\n  num_threads: 1\n", encoding="utf-8")
    (resource_reader / "fixture.json").write_text(json.dumps({"roots": [str(resource_farm)]}), encoding="utf-8")
    resource_adapter = ["--chia-root", resource_reader, "--chia-executable", executable]
    resource_inventory = directory / "resource-only inventory.plan"
    resource_catalog = json.loads(run("plan", "--inventory", resource_farm, *resource_adapter,
                                      "--save-plan", resource_inventory, "--json").stdout)
    resource_profile = json.loads(json.dumps(profile))
    for workload in resource_profile["farm_resources"]["workloads"]:
        for use in workload["uses"]:
            if use["resource"] == cpu:
                use.update(quality_us=10000000, proof_us=10000000)
    for phase in ("construction", "verification"):
        for demand in resource_profile["farm_resources"]["native"][phase]:
            if demand["resource"] == cpu:
                demand["work_us"] = 55000000
    resource_file = directory / "resource-only profile.json"
    resource_file.write_text(json.dumps(resource_profile), encoding="utf-8")
    before = snapshot(resource_farm)
    refused = json.loads(run("plan", resource_farm, "--replot", resource_inventory, "--harvester", resource_file,
                            *resource_adapter, *options, "--json", code=2).stdout)
    assert not refused["feasible"] and any("response-deadline-exceeded" in reason for reason in refused["infeasibility_reasons"])
    refused_file = directory / "infeasible resource review.plan"
    refused_saved = json.loads(run("plan", resource_farm, "--replot", resource_inventory, "--harvester", resource_file,
                                  *resource_adapter, *options, "--save-plan", refused_file, "--json", code=2).stdout)
    assert refused_saved["infeasibility_reasons"] == refused["infeasibility_reasons"] and not refused_saved["saved_plan"]
    assert not refused_file.exists() and snapshot(resource_farm) == before, "infeasible saved review changed the farm"
    resource_credit = resource_catalog["plots"][0]["identity"]["allocated_bytes"]
    resource_review = directory / "resource-only migration.plan"
    reviewed_resource = json.loads(run("plan", resource_farm, "--replot", resource_inventory, "--harvester", resource_file,
                                      *resource_adapter, *options, "--reclaim-before-bytes", resource_credit,
                                      "--save-plan", resource_review, "--json").stdout)
    assert reviewed_resource["feasible"] and snapshot(resource_farm) == before
    import shutil
    assert shutil.disk_usage(resource_farm).free > ram + 256 * mib, "resource-only fixture unexpectedly depends on disk pressure"
    resource_final = Path(reviewed_resource["batches"][0]["path"])
    run("fill", resource_review, "--harvester", resource_file, *resource_adapter, *options, *keys,
        point="replot-source-unlinked", code=86)
    assert not resource_source.exists() and not resource_final.exists()
    assert not list(resource_farm.glob(".xchplot2-stage-*.building")), "resource-only predelete followed construction instead of enabling it"
    unlinked = json.loads(run("plan", "--status", "--from-plan", resource_review, "--json").stdout)
    assert unlinked["active_sources"][0]["durable_state"] == "claimed"
    assert unlinked["active_sources"][0]["current_native_observation"] == "original-and-claim-absent"
    assert unlinked["outstanding_predelete_bytes"] == unlinked["predelete_limit_bytes"] == resource_credit
    resource_private_farm = private_replot_farm(resource_farm / "grow-replot.plan")
    # Review a changed native phase envelope after unlink but before retirement
    # is recorded. The old workload bounds remain; current unconfirmed demand
    # still fits, while the original physical deletion debt cannot disappear.
    unlinked_profile = json.loads(json.dumps(resource_profile))
    for phase in ("construction", "verification"):
        original_work = next(demand["work_us"] for demand in profile["farm_resources"]["native"][phase] if demand["resource"] == cpu)
        for demand in unlinked_profile["farm_resources"]["native"][phase]:
            if demand["resource"] == cpu:
                demand["work_us"] = original_work
    unlinked_profile_file = directory / "unlinked successor profile.json"
    unlinked_profile_file.write_text(json.dumps(unlinked_profile), encoding="utf-8")
    unlinked_successor_file = directory / "unlinked successor.plan"
    before = snapshot(resource_farm)
    unlinked_successor = json.loads(run("plan", "--revalidate", "--from-plan", resource_review, "--harvester", unlinked_profile_file,
                                       *resource_adapter, *options, "--save-plan", unlinked_successor_file, "--json").stdout)
    assert snapshot(resource_farm) == before and not unlinked_successor["activated"]
    run("fill", "--from-plan", unlinked_successor_file, "--harvester", unlinked_profile_file, *resource_adapter, *options,
        point="replot-policy-activated", code=86)
    debt = json.loads(run("plan", "--status", "--from-plan", unlinked_successor_file, "--json").stdout)
    assert debt["selected_policy_binding"] == "active" and debt["active_sources"][0]["durable_state"] == "claimed"
    assert debt["outstanding_predelete_bytes"] == debt["predelete_limit_bytes"] == resource_credit
    assert private_replot_farm(resource_farm / "grow-replot.plan") == resource_private_farm
    before = snapshot(resource_farm)
    run("fill", "--from-plan", resource_review, *options, code=2)
    assert snapshot(resource_farm) == before, "original review changed an activated unlink-before-retirement successor"
    run("fill", "--from-plan", unlinked_successor_file, *options)
    assert digest(resource_final) == replacement_hash and (resource_farm / "grow-farm.plan").is_file()
    complete = json.loads(run("plan", "--status", "--from-plan", unlinked_successor_file, "--json").stdout)
    assert complete["outstanding_predelete_bytes"] == 0 and complete["completed_receipt_chain_verified"]
    assert private_replot_farm(resource_farm / "grow-replot.plan") == resource_private_farm
    resource_receipt = resource_final.parent / f".xchplot2-replot-{reviewed_resource['job_id']}" / "receipt-0"
    completed_output, = complete["completed_outputs"]
    assert completed_output["current_native_observation"] == f"matching-saved-native-{saved_replacement_authority(resource_receipt)}"
    assert shlex.split(resource_receipt.read_bytes()[:-65].decode("utf-8"))[1] == unlinked_successor["origin_review_digest"]
    proof = run("verify", resource_final, "--full", "--trials", 100)
    found = re.search(r"(\d+) full proofs validated", (proof.stdout + proof.stderr).lower())
    assert found and int(found[1]) > 0
    print("PASS resource-only predelete despite ample disk, exact source unload, post-unlink recovery, byte parity and full proofs", flush=True)

    # Key-free review does not prescribe NFT versus OG. Once prepared, keys
    # are immutable, as checked above; a first preparation may choose OG.
    og_farm = directory / "OG migration"
    og_farm.mkdir()
    og_source = og_farm / "legacy.plot"
    og_source.write_bytes(legacy)
    og_reader = directory / "OG reader"
    (og_reader / "config").mkdir(parents=True)
    (og_reader / "config" / "config.yaml").write_text("self_hostname: localhost\nharvester:\n  num_threads: 1\n", encoding="utf-8")
    (og_reader / "fixture.json").write_text(json.dumps({"roots": [str(og_farm)]}), encoding="utf-8")
    og_adapter = ["--chia-root", og_reader, "--chia-executable", executable]
    og_inventory = directory / "OG inventory.plan"
    run("plan", "--inventory", og_farm, *og_adapter, "--save-plan", og_inventory, "--json")
    og_review = directory / "OG migration.plan"
    reviewed_og = json.loads(run("plan", og_farm, "--replot", og_inventory, "--harvester", profile_file,
                                *og_adapter, *options, "--save-plan", og_review, "--json").stdout)
    run("fill", og_review, "--harvester", profile_file, *og_adapter, *options,
        "-f", FARMER, "--pool-pk", FARMER, "-S", SEED)
    og_final = Path(reviewed_og["batches"][0]["path"])
    assert not og_source.exists() and len(header(og_final)[5]) == 256
    proof = run("verify", og_final, "--full", "--trials", 100)
    found = re.search(r"(\d+) full proofs validated", (proof.stdout + proof.stderr).lower())
    assert found and int(found[1]) > 0
    print("PASS first preparation OG route, truthful loaded key metadata and full proofs", flush=True)

    # A real saved pool and embedded CPU executor use the same migration
    # controller. These declared protocol fixtures do not measure farm load.
    cpus = len(os.sched_getaffinity(0))
    if cpus < 4:
        print("SKIP protected compute migration: insufficient inherited CPU lanes for the complete concurrent fixture reservations", flush=True)
        return
    compute_farm = directory / "compute migration"
    compute_farm.mkdir()
    compute_source = compute_farm / "legacy.plot"
    compute_source.write_bytes(legacy)
    compute_reader = directory / "compute reader"
    (compute_reader / "config").mkdir(parents=True)
    (compute_reader / "config" / "config.yaml").write_text("self_hostname: localhost\nharvester:\n  num_threads: 1\n", encoding="utf-8")
    (compute_reader / "fixture.json").write_text(json.dumps({"roots": [str(compute_farm)],
                                                            "plot_public_key": COMPUTE_FARMING_PLOT_KEY}), encoding="utf-8")
    compute_adapter = ["--chia-root", compute_reader, "--chia-executable", executable]
    compute_inventory = directory / "compute inventory.plan"
    compute_catalog = json.loads(run("plan", "--inventory", compute_farm, *compute_adapter,
                                     "--save-plan", compute_inventory, "--json").stdout)
    membership = directory / "migration compute membership"
    run("compute", "init", "--directory", membership, "--name", "protected migration fixture")
    # Leave saved caps unset so the explicit offers below reach the reviewed
    # physical envelope rather than being clipped by membership defaults.
    run("compute", "configure", "--directory", membership, "--interface", "lo", "--discovery", "off",
        "--backend", "cpu", "--cpus", 0, "--host-bytes", 0, "--max-brokers", 16)
    import ssl
    credentials = membership / "credentials.pem"
    if not credentials.exists():
        credentials = membership / "cert.pem"
    certificate = re.search(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
                            credentials.read_text(encoding="utf-8"), re.S).group()
    identity = hashlib.sha256(ssl.PEM_cert_to_DER_cert(certificate)).hexdigest()
    compute_profile = json.loads(json.dumps(profile))
    compute_resources = compute_profile["farm_resources"]
    for resource in compute_resources["resources"]:
        if resource["kind"] == "cpu":
            resource["capacity"] = cpus
    for workload, current in zip(compute_resources["workloads"], compute_catalog["farm_resources"]["workloads"]):
        workload.update(id=current["id"], plot_count=current["plot_count"])
    native = compute_resources["native"]
    native.update(engine="external", max_ram_bytes=2 * ram, max_batch_members=2, max_group_members=2,
                  scratch_resources=disks, construction=demands(2 * ram + 64 * mib, phase_cpu),
                  verification=demands(2 * ram + 64 * mib, phase_cpu),
                  compute_owner_identity=identity, compute_broker_identity=identity, compute_broker_host=compute_catalog["host"])
    native["compute_services"] = demands(192 * mib)
    native["compute_services"][-1].update(resident_units=128 * mib, transient_units=64 * mib)
    native["compute_workers"] = [{"tls_identity": identity, "host": compute_catalog["host"], "gpu_identity": "",
                                  "cpu_threads": 2, "slots": 1, "host_bytes": ram // 2, "gpu_bytes": 0,
                                  "standing_host_bytes": 256 * mib, "standing_gpu_bytes": 0,
                                  "allowed_kinds": ["generate-cpu", "encode"],
                                  "reservation": [{"resource": cpu, "reserved_units": 2, "bounded": True},
                                                  {"resource": memory, "resident_units": 256 * mib,
                                                   "transient_units": ram // 2 - 256 * mib, "bounded": True}]}]
    compute_file = directory / "protected compute profile.json"
    compute_file.write_text(json.dumps(compute_profile), encoding="utf-8")
    compute_options = list(external_options)
    for key, value in (("--compute", "auto"), ("--group-members", 2), ("--max-total-members", 2),
                       ("--batch-members", 2), ("-M", 2 * ram), ("--capacity-bytes", 2 * ram + 256 * mib)):
        compute_options[compute_options.index(key) + 1] = value
    compute_options += ["--compute-directory", membership, "--compute-owner-id", "protected-migration-owner",
                        "--compute-offer-ram", "1G", "--compute-offer-cpus", 2, "--compute-offer-slots", 1,
                        "--compute-offer-backend", "cpu", "--compute-codec-ram", "64M", "--compute-inflight", 1,
                        "--compute-timeout-ms", 120000]
    ordinary = directory / "ordinary shared scheduler"
    ordinary.mkdir()
    refused_ordinary = run("plan", ordinary, *compute_options, "--json", code=2)
    assert "local compute offer must leave cpu capacity" in (refused_ordinary.stdout + refused_ordinary.stderr).lower()
    assert not list(ordinary.iterdir()), "ordinary shared-scheduler refusal changed its destination"
    compute_review = directory / "protected compute.plan"
    before = snapshot(compute_farm)
    uncovered_profile = json.loads(json.dumps(compute_profile))
    for field in ("compute_owner_identity", "compute_broker_identity", "compute_broker_host", "compute_services", "compute_workers"):
        uncovered_profile["farm_resources"]["native"].pop(field)
    uncovered_file = directory / "uncovered compute profile.json"
    uncovered_file.write_text(json.dumps(uncovered_profile), encoding="utf-8")
    uncovered = json.loads(run("plan", compute_farm, "--replot", compute_inventory, "--harvester", uncovered_file,
                              *compute_adapter, *compute_options, "--json", code=2).stdout)
    assert not uncovered["feasible"] and any("compute requires reviewed" in reason for reason in uncovered["infeasibility_reasons"])
    assert snapshot(compute_farm) == before
    compute_credit = compute_catalog["plots"][0]["identity"]["allocated_bytes"]
    reviewed_compute = json.loads(run("plan", compute_farm, "--replot", compute_inventory, "--harvester", compute_file,
                                     *compute_adapter, *compute_options, "--reclaim-before-bytes", compute_credit,
                                     "--save-plan", compute_review, "--json").stdout)
    assert reviewed_compute["feasible"] and reviewed_compute["native_request"]["compute_enabled"] and snapshot(compute_farm) == before
    huge_offer = list(compute_options)
    huge_offer[huge_offer.index("--compute-offer-cpus") + 1] = 1025
    refused_huge = run("fill", compute_review, "--harvester", compute_file, *compute_adapter, *huge_offer, *keys, code=2)
    assert "local compute offer exceeds CPU or slot bounds" in refused_huge.stdout + refused_huge.stderr
    assert snapshot(compute_farm) == before and not (compute_farm / "grow-replot.plan").exists()
    bad_offer = list(compute_options)
    bad_offer[bad_offer.index("--compute-offer-ram") + 1] = "2G"
    refusal = run("fill", compute_review, "--harvester", compute_file, *compute_adapter, *bad_offer, *keys, code=2)
    assert "local compute offer exceeds" in refusal.stdout + refusal.stderr
    assert snapshot(compute_farm) == before and not (compute_farm / "grow-replot.plan").exists(), "bad local offer claimed a source before executor admission"
    run("fill", compute_review, "--harvester", compute_file, *compute_adapter, *compute_options, *keys,
        point="replot-journal-all-staged", code=86)
    run("fill", compute_review, "--harvester", compute_file, *compute_adapter, *compute_options,
        point="replot-replacement-verified", code=86)
    run("fill", compute_review, "--harvester", compute_file, *compute_adapter, *compute_options)
    compute_final = Path(reviewed_compute["batches"][0]["path"])
    assert not compute_source.exists() and header(compute_final)[3] == 2
    proof = run("verify", compute_final, "--full", "--trials", 100)
    found = re.search(r"(\d+) full proofs validated", (proof.stdout + proof.stderr).lower())
    assert found and int(found[1]) > 0
    if reference:
        group, k, strength, count, meta, memo = header(compute_final)
        raw_directory = directory / "compute reference fragments"
        raw_directory.mkdir()
        raw = []
        for index in range(count):
            path = raw_directory / f"{index}.plot2"
            run("test", k, group, strength, index, meta, "--raw", "-m", memo, "-o", raw_directory, "-N", path.name)
            raw.append(path)
        inputs = raw_directory / "group.inputs"
        inputs.write_text(f"{k} {strength} {meta} {group} {memo}\n" +
                          "".join(json.dumps(str(path), ensure_ascii=False) + "\n" for path in raw), encoding="utf-8")
        canonical = raw_directory / "canonical.gplot"
        comparison = subprocess.run([str(reference), "assemble-only", str(inputs), str(canonical), "512"],
                                    capture_output=True, text=True, encoding="utf-8", timeout=180)
        assert comparison.returncode == 0, (comparison.stdout, comparison.stderr)
        assert digest(canonical) == digest(compute_final), "protected compute migration differs from the independent assembler"
    print("PASS saved-pool protected migration, complete worker reservations, preclaim offer refusal, owner-key recovery, real multi-member contributions and full proofs", flush=True)


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
