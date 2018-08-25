#!/bin/sh

set -ex

cargo build --verbose
cargo doc --verbose
cargo test --verbose
if [ "$TRAVIS_RUST_VERSION" = "nightly" ]; then
  cargo bench --verbose --no-run
fi
cargo build --verbose --manifest-path=stree_cmd/Cargo.toml
cargo build --verbose --manifest-path=suffix_tree/Cargo.toml
cargo test --verbose --manifest-path=suffix_tree/Cargo.toml
cargo doc --verbose --manifest-path=suffix_tree/Cargo.toml
