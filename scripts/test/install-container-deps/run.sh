#!/usr/bin/env bash
#
# run.sh — verify install-container-deps.sh's --dry-run output matches
# checked-in fixtures across (distro × engine × gpu) combinations.
#
# Each distro's full (engine × gpu) matrix runs inside a single
# arch/ubuntu/fedora container, so the cost is three image pulls + three
# container startups regardless of how many tuples the matrix expands to.
#
# Usage:
#   scripts/test/install-container-deps/run.sh            # diff mode (CI default)
#   scripts/test/install-container-deps/run.sh --update   # regenerate fixtures
#
# Honours $XCHPLOT2_CONTAINER_RUNTIME (podman|docker); auto-detects
# otherwise, preferring podman.

set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
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

    # All (engine × gpu) combos for this distro run in one container.
    # Each combo gets a `=== engine=X gpu=Y ===` header so the fixture
    # diffs cleanly when one tuple drifts.
    # shellcheck disable=SC2016  # $engine/$gpu intentionally evaluated inside the container shell
    "$RUNTIME" run --rm -v "$ROOT/scripts:/s:ro" "$img" bash -c '
        for engine in podman docker; do
            for gpu in nvidia amd intel cpu; do
                printf "=== engine=%s gpu=%s ===\n" "$engine" "$gpu"
                /s/install-container-deps.sh --dry-run \
                    --engine "$engine" --gpu "$gpu" 2>&1 \
                    || printf "[exit=%d]\n" $?
                printf "\n"
            done
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
