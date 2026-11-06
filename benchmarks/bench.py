"""Benchmark OptiSpec models and compare runs.

Usage:
    uv run benchmarks/bench.py run --tag NAME [--x64] [--quick] [--repeats N] [--cases A B ...] [--force]
    uv run benchmarks/bench.py compare BASELINE NEW [--tolerance X]

`run` writes benchmarks/results/<tag>-<f32|f64>.json (timings, memory) and .npz
(spectra). `compare` takes two of those paths without the extension and exits 1 if
a case is missing or a spectrum changed beyond the tolerance. Speed never fails it.

`--quick` runs only the small cases, once each, for fast feedback while changing
the models (about 10s instead of 40s in f32). Compare quick runs with quick runs.

Don't run other GPU work during a benchmark: it skews timings and memory.
"""

import argparse
import dataclasses
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from optispec.models import mlj, stark, two_state
from optispec.models.base import Spectrum

RESULTS_DIR = Path(__file__).resolve().parent / "results"

Precision = Literal["f32", "f64"]
FloatArray = NDArray[np.floating]

DEFAULT_TOLERANCE: dict[Precision, float] = {"f32": 1e-5, "f64": 1e-10}


# Every parameter is pinned so a change to a library default can't silently change
# the workload.
def mlj_spectrum() -> Spectrum:
    return mlj.absorption(
        mlj.Params(
            start_energy=0.0,
            end_energy=20_000.0,
            num_points=2_001,
            basis_size=10,
            temperature_kelvin=300.0,
            energy_gap=8_000.0,
            disorder_meV=0.0,
            mode_frequencies=jnp.array([1200.0, 100.0]),
            mode_couplings=jnp.array([0.7, 2.0]),
        )
    )


def two_state_params(basis: tuple[int, int]) -> two_state.Params:
    return two_state.Params(
        start_energy=0.0,
        end_energy=20_000.0,
        num_points=2_001,
        temperature_kelvin=300.0,
        broadening=200.0,
        coupling=100.0,
        energy_gap=8_000.0,
        mode_basis_sets=basis,
        mode_frequencies=jnp.array([1200.0, 100.0]),
        mode_couplings=jnp.array([0.7, 2.0]),
    )


def two_state_spectrum(basis: tuple[int, int]) -> Spectrum:
    return two_state.absorption(two_state_params(basis))


def stark_spectrum(basis: tuple[int, int]) -> Spectrum:
    return stark.absorption(
        stark.Params(
            model=two_state,
            neutral_params=two_state_params(basis),
            field_strength=0.01,
            positive_field_contribution_ratio=0.5,
            field_delta_dipole=38.0,
            field_delta_polarizability=1000.0,
        )
    )


CASES: dict[str, Callable[[], Spectrum]] = {
    "mlj": mlj_spectrum,
    "two_state_20x50": partial(two_state_spectrum, (20, 50)),
    "stark_20x50": partial(stark_spectrum, (20, 50)),
    "two_state_20x200": partial(two_state_spectrum, (20, 200)),
    "stark_20x200": partial(stark_spectrum, (20, 200)),
}
QUICK_CASES = ("mlj", "two_state_20x50", "stark_20x50")


@dataclasses.dataclass(frozen=True, slots=True)
class CaseResult:
    first_call_s: float
    samples_s: tuple[float, ...]
    peak_device_bytes: int | None

    @property
    def median_s(self) -> float:
        return statistics.median(self.samples_s)


@dataclasses.dataclass(frozen=True, slots=True)
class RunInfo:
    precision: Precision
    device: str
    jax_version: str
    repeats: int


@dataclasses.dataclass(frozen=True, slots=True)
class Measurement:
    result: CaseResult
    energies: FloatArray
    intensities: FloatArray
    device: str


@dataclasses.dataclass(frozen=True, slots=True)
class SavedRun:
    info: RunInfo
    results: dict[str, CaseResult]
    arrays: dict[str, FloatArray]


def precision_of(x64: bool) -> Precision:
    return "f64" if x64 else "f32"


def result_paths(stem: Path) -> tuple[Path, Path]:
    # Not with_suffix: a tag like "v1.2" would lose its ".2".
    return stem.with_name(f"{stem.name}.json"), stem.with_name(f"{stem.name}.npz")


def format_bytes(n: int | None) -> str:
    return "n/a" if n is None else f"{n / 1e9:.2f}GB"


def measure(case: str, x64: bool, repeats: int) -> Measurement:
    """Time one case. Run it in a fresh process so memory and compile time are per case."""
    # Must happen before this process creates any JAX array.
    jax.config.update("jax_enable_x64", x64)
    build = CASES[case]

    def timed_call() -> tuple[float, Spectrum]:
        start = time.perf_counter()
        spectrum = build()
        jax.block_until_ready(spectrum)
        return time.perf_counter() - start, spectrum

    first_call_s, spectrum = timed_call()
    samples = []
    for _ in range(repeats):
        seconds, spectrum = timed_call()
        samples.append(seconds)

    device = jax.devices()[0]
    peak = (device.memory_stats() or {}).get("peak_bytes_in_use")
    return Measurement(
        result=CaseResult(first_call_s, tuple(samples), peak),
        energies=np.asarray(spectrum.energies),
        intensities=np.asarray(spectrum.intensities),
        device=f"{device.platform}/{device.device_kind}",
    )


def measure_in_fresh_process(case: str, x64: bool, repeats: int) -> Measurement:
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        return pool.submit(measure, case, x64, repeats).result()


def spectrum_change(base: FloatArray, new: FloatArray) -> tuple[float, float]:
    """Return (largest difference after scaling each to its peak, ratio of peaks).

    The ratio is needed because peak scaling hides a spectrum that is uniformly off.
    NaN or all-zero spectra produce NaN, which fails any tolerance check.
    """
    with np.errstate(all="ignore"):
        base_peak, new_peak = np.abs(base).max(), np.abs(new).max()
        drift = np.abs(base / base_peak - new / new_peak).max()
        return float(drift), float(new_peak / base_peak)


def save_run(stem: Path, run: SavedRun) -> None:
    json_path, npz_path = result_paths(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    # numpy's stubs can't rule out a key named "allow_pickle" in the unpacked dict.
    np.savez_compressed(npz_path, **run.arrays)  # type: ignore[arg-type]
    data = {
        "info": dataclasses.asdict(run.info),
        "results": {case: dataclasses.asdict(r) for case, r in run.results.items()},
    }
    json_path.write_text(json.dumps(data, indent=2) + "\n")


def load_run(stem: Path) -> SavedRun:
    json_path, npz_path = result_paths(stem)
    data = json.loads(json_path.read_text())
    results = {
        case: CaseResult(**{**r, "samples_s": tuple(r["samples_s"])})
        for case, r in data["results"].items()
    }
    with np.load(npz_path) as npz:
        arrays = {key: npz[key] for key in npz.files}
    return SavedRun(RunInfo(**data["info"]), results, arrays)


def run(tag: str, x64: bool, repeats: int, cases: Sequence[str], force: bool) -> int:
    precision = precision_of(x64)
    stem = RESULTS_DIR / f"{tag}-{precision}"
    if not force and any(path.exists() for path in result_paths(stem)):
        sys.exit(f"error: {stem.name} already exists (use --force to overwrite)")

    results: dict[str, CaseResult] = {}
    arrays: dict[str, FloatArray] = {}
    devices: set[str] = set()
    for case in cases:
        m = measure_in_fresh_process(case, x64, repeats)
        results[case] = m.result
        arrays[f"{case}/energies"] = m.energies
        arrays[f"{case}/intensities"] = m.intensities
        devices.add(m.device)
        print(
            f"{case:18s} first={m.result.first_call_s:7.3f}s  "
            f"median={m.result.median_s:7.3f}s  "
            f"range={min(m.result.samples_s):.3f}-{max(m.result.samples_s):.3f}s  "
            f"peak_mem={format_bytes(m.result.peak_device_bytes)}",
            flush=True,
        )

    info = RunInfo(
        precision=precision,
        device=", ".join(sorted(devices)),
        jax_version=jax.__version__,
        repeats=repeats,
    )
    save_run(stem, SavedRun(info, results, arrays))
    print(f"wrote {stem}.{{json,npz}}: {info}")
    return 0


def compare(baseline: Path, new: Path, tolerance: float | None) -> int:
    base_run, new_run = load_run(baseline), load_run(new)
    base_info, new_info = base_run.info, new_run.info
    if base_info.precision != new_info.precision:
        sys.exit(
            f"error: precision differs: {base_info.precision} vs {new_info.precision}"
        )
    if base_info.device != new_info.device:
        sys.exit(f"error: device differs: {base_info.device} vs {new_info.device}")
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCE[base_run.info.precision]

    print(f"baseline: {base_run.info}\nnew:      {new_run.info}\n")
    print(
        f"{'case':18s} {'base':>8s} {'new':>8s} {'speedup':>8s} {'base 1st':>9s} "
        f"{'new 1st':>9s} {'base mem':>9s} {'new mem':>9s} {'drift':>9s} "
        f"{'scale':>11s}  status"
    )
    failed = False
    for case in dict.fromkeys([*base_run.results, *new_run.results]):
        base, new_result = base_run.results.get(case), new_run.results.get(case)
        if base is None or new_result is None:
            failed = True
            print(
                f"{case:18s} FAIL: missing in {'new' if base is not None else 'baseline'}"
            )
            continue

        problems: list[str] = []
        drift = scale = float("nan")
        base_energies = base_run.arrays[f"{case}/energies"]
        if not np.array_equal(base_energies, new_run.arrays[f"{case}/energies"]):
            problems.append("energy grids differ")
        else:
            drift, scale = spectrum_change(
                base_run.arrays[f"{case}/intensities"],
                new_run.arrays[f"{case}/intensities"],
            )
            # "not <=" rather than ">" so that NaN fails too.
            if not drift <= tolerance:
                problems.append(f"drift {drift:.1e}")
            if not abs(scale - 1) <= tolerance:
                problems.append(f"scale {scale:.6g}")
        failed = failed or bool(problems)

        status = f"FAIL: {', '.join(problems)}" if problems else "ok"
        print(
            f"{case:18s} {base.median_s:7.3f}s {new_result.median_s:7.3f}s "
            f"{base.median_s / new_result.median_s:7.2f}x "
            f"{base.first_call_s:8.3f}s {new_result.first_call_s:8.3f}s "
            f"{format_bytes(base.peak_device_bytes):>9s} "
            f"{format_bytes(new_result.peak_device_bytes):>9s} "
            f"{drift:9.2e} {scale:11.8f}  {status}"
        )

    print(f"\ntolerance {tolerance:g}")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="benchmark the cases and save results")
    run_parser.add_argument("--tag", required=True, help="name for the result files")
    run_parser.add_argument("--x64", action="store_true", help="use 64-bit floats")
    run_parser.add_argument(
        "--quick", action="store_true", help="fast feedback: small cases, one repeat"
    )
    run_parser.add_argument(
        "--repeats",
        type=int,
        help="timed calls per case (default 5, or 1 with --quick)",
    )
    run_parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        help=f"default: all, or {' '.join(QUICK_CASES)} with --quick",
    )
    run_parser.add_argument("--force", action="store_true", help="overwrite results")

    compare_parser = commands.add_parser("compare", help="compare two saved runs")
    compare_parser.add_argument("baseline", type=Path, help="e.g. results/before-f32")
    compare_parser.add_argument("new", type=Path, help="e.g. results/after-f32")
    compare_parser.add_argument(
        "--tolerance",
        type=float,
        help="max spectrum change (default 1e-5 for f32, 1e-10 for f64)",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        repeats = args.repeats if args.repeats is not None else 1 if args.quick else 5
        if repeats < 1:
            parser.error("--repeats must be at least 1")
        cases = args.cases or (QUICK_CASES if args.quick else list(CASES))
        return run(args.tag, args.x64, repeats, cases, args.force)
    return compare(args.baseline, args.new, args.tolerance)


if __name__ == "__main__":
    sys.exit(main())
