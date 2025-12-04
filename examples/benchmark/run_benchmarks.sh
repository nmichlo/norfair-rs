#!/bin/bash

uv run generate_data.py

function _test() {
    uv run benchmark_python.py $1
    go run benchmark_go.go $1
    RUSTFLAGS=-Awarnings cargo run --release --example benchmark_rust $1
}

_test small
_test medium
_test large
_test stress
