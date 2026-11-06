# Benchmarks

`bench.py` times the models, records peak GPU memory, and checks that changes don't alter the output spectra.

## Usage

```zsh
uv run benchmarks/bench.py run --tag before          # f32, all cases, about 40s
uv run benchmarks/bench.py run --tag before --x64    # f64, about 3 min
uv run benchmarks/bench.py run --tag before --quick  # small cases once each, about 10s

uv run benchmarks/bench.py compare benchmarks/results/before-f32 benchmarks/results/after-f32
```

`run` writes `results/<tag>-<f32|f64>.json` and `.npz`. It refuses to overwrite an existing tag unless you pass `--force`. Use `--cases` to pick cases and `--repeats` to set the number of timed calls (default 5, or 1 with `--quick`).

`compare` takes paths without the extension. It exits 1 if a case is missing or a spectrum changed by more than the tolerance (1e-5 in f32, 1e-10 in f64, set with `--tolerance`). It refuses to compare runs with different precision or devices. Speed differences never fail it.

## Notes

- Each case runs in a fresh process, so its first-call time and peak memory don't depend on the other cases.
- Compare f32 runs with f32 runs and quick runs with quick runs.
- Don't run other GPU work during a benchmark. Two JAX processes on one GPU slow each other down or run out of memory.
- `results/` is gitignored.
