#!/bin/bash

function _py() {
    uv run benchmark_python.py $1
}
function _py_rs() {
    uv run python benchmark_python.py $1 --norfair-rs
}
function _go() {
    go run benchmark_go.go $1
}
function _rs() {
    RUSTFLAGS=-Awarnings cargo run --quiet --release --manifest-path "$_repo_root/Cargo.toml" --example benchmark_rust $1
}

echo ; echo "- - - SETUP - - -"

# get script directory and repo root
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="$(cd "$_script_dir/../.." && pwd)"
# change to script directory for relative paths
cd "$_script_dir"
# generate data
uv run generate_data.py
# generate norfair_rs python lib
(cd "$_repo_root" && RUSTFLAGS=-Awarnings uv run maturin develop --release --quiet )

echo ; echo "===== WARMUP ====="
_rs small ; _rs small
_py_rs small ; _py_rs small
_go small ; _go small
_py small ; _py small

echo ; echo "===== RUST ====="
_rs small
_rs medium
_rs large
_rs stress

echo ; echo "===== PYTHON (norfair_rs) ====="
_py_rs small
_py_rs medium
_py_rs large
_py_rs stress

echo ; echo "===== GO ====="
_go small
_go medium
_go large
_go stress

echo ; echo "===== PYTHON ====="
_py small
_py medium
_py large
echo skipped python:stress takes long
# _py stress
