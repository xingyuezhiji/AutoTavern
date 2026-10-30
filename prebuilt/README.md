# Linux simulator wheel

This wheel contains the public TavernGym simulator from this checkout.
It contains no policy weights, opponent state, or credentials.

Its SHA256 and source fingerprint are recorded in `openi_taverngym_wheel.json`.
The training entry verifies both before using it. It supports Linux x86_64
and Python 3.10 or newer through the CPython stable ABI.

Build with Rust 1.98.1, maturin 1.15.0, and Zig 0.14.1:

```sh
maturin build --release --zig --target x86_64-unknown-linux-gnu \
  --compatibility manylinux_2_17 --out /tmp/autotavern-linux-wheels
```

Verified in a Linux x86_64 Python 3.12 container: source fingerprint
`6d8712c93022ba11`, 4017 float observation features, and a complete
eight-player scripted game with 1076 actions.
