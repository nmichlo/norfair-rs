#!/bin/bash

uv run generate_data.py

function _py() {
    uv run benchmark_python.py $1
}
function _go() {
    go run benchmark_go.go $1
}
function _rs() {
    RUSTFLAGS=-Awarnings cargo run --quiet --release --example benchmark_rust $1
}

echo ; echo "===== RUST ====="
_rs small
_rs medium
_rs large
_rs stress

echo ; echo "===== GO ====="
_go small
_go medium
_go large
_go stress

echo ; echo "===== PYTHON ====="
_py small
_py medium
_py large
echo skipped python:stress
# _py stress

