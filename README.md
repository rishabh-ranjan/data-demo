# data-demo

`data-rs` contains the Rust code.

`data-rs/src/pre.rs` contains pre-processing code (produces pre-processed files in ~/scratch/pre).
`data-rs/src/fly.rs` contains the code for on-the-fly data generation (randomized BFS).

`data.py` contains the PyTorch DataLoader which calls the Rust code internally.
