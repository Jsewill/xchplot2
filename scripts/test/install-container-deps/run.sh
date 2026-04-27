#!/usr/bin/env bash
#
# run.sh — verify install-container-deps.sh's --dry-run output matches
# checked-in fixtures across (distro × engine) combinations.
#
# cuda-only is NVIDIA-only so there's no `--gpu` axis to vary; the
# matrix is just (distro, engine).
#
# Each distro's full engine matrix runs inside a single arch / ubuntu /
# fedora container, so the cost is three image pulls + three container
# startups.
#
# Usage:
#   scripts/test/install-container-deps/run.sh            # diff mode (CI default)
#   scripts/test/install-container-deps/run.sh --update   # regenerate fixtures
#
# Honours $XCHPLOT2_CONTAINER_RUNTIME (podman|docker); auto-detects
# otherwise, preferring podman.

set -euo pipefail

# Derive ROOT from this script's own path so the harness works no
# matter what CWD it runs from. The previous `git rev-parse` form
# resolved against the *outer* CWD, so running this script from
# another repo's directory wrote fixtures into the wrong tree.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
FIXTURE_DIR="$ROOT/scripts/test/install-container-deps"

UPDATE=0
[[ "${1:-}" == --update ]] && UPDATE=1

if [[ -n "${XCHPLOT2_CONTAINER_RUNTIME:-}" ]]; then
    RUNTIME="$XCHPLOT2_CONTAINER_RUNTIME"
elif command -v podman >/dev/null; then
    RUNTIME=podman
elif command -v docker >/dev/null; then
    RUNTIME=docker
else
    echo "run.sh: neither podman nor docker on PATH" >&2
    exit 1
fi

declare -A IMAGES=(
    [arch]=docker.io/archlinux:latest
    [ubuntu]=docker.io/ubuntu:24.04
    [fedora]=docker.io/fedora:40
)

# `XCHPLOT2_DRY_DISTRO_FILTER=arch` runs only one distro — handy when
# regenerating a single fixture without re-pulling all three images.
FILTER="${XCHPLOT2_DRY_DISTRO_FILTER:-}"

failed=0
for distro in arch ubuntu fedora; do
    [[ -z "$FILTER" || "$FILTER" == "$distro" ]] || continue

    img="${IMAGES[$distro]}"
    fixture="$FIXTURE_DIR/$distro.txt"
    tmp=$(mktemp)
    # shellcheck disable=SC2064  # intentional early expansion
    trap "rm -f '$tmp'" EXIT

    # Both engines run in one container; each gets a `=== engine=X ===`
    # header so the fixture diffs cleanly when one tuple drifts.
    # shellcheck disable=SC2016  # $engine intentionally evaluated inside the container shell
    "$RUNTIME" run --rm -v "$ROOT/scripts:/s:ro" "$img" bash -c '
        for engine in podman docker; do
            printf "=== engine=%s ===\n" "$engine"
            /s/install-container-deps.sh --dry-run --engine "$engine" 2>&1 \
                || printf "[exit=%d]\n" $?
            printf "\n"
        done
    ' > "$tmp"

    if (( UPDATE )); then
        cp "$tmp" "$fixture"
        echo "updated: $fixture"
    elif ! diff -u "$fixture" "$tmp"; then
        echo "::error::fixture mismatch for distro=$distro"
        failed=1
    else
        echo "ok: $distro"
    fi
done

exit $failed
