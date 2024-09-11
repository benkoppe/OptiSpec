import pytest
import jax.numpy as jnp

from optispec.models import two_state as ts
from tests.data.two_state import OLD_PACKAGE_ABSORPTION, FORTRAN_CODE_ABSORPTION


@pytest.fixture
def two_state_params():
    return ts.Params(
        start_energy=0.0,
        end_energy=20_000.0,
        num_points=2_001,
        temperature_kelvin=300.0,
        broadening=200.0,
        transfer_integral=100.0,
        energy_gap=8_000.0,
        mode_basis_sets=(20, 50),
        mode_frequencies=jnp.array([1200.0, 100.0]),
        mode_couplings=jnp.array([0.7, 2.0]),
    )


def test_two_state_absorption_against_old_python_package(two_state_params):
    params = two_state_params
    absorption = ts.absorption(params)

    absorption = absorption.match_greatest_peak_of(OLD_PACKAGE_ABSORPTION.intensities)
    assert absorption.assert_similarity(OLD_PACKAGE_ABSORPTION)


def test_two_state_absorption_against_fortran_code(two_state_params):
    params = two_state_params
    absorption = ts.absorption(params)

    absorption = absorption.match_greatest_peak_of(FORTRAN_CODE_ABSORPTION.intensities)
    assert absorption.assert_similarity(FORTRAN_CODE_ABSORPTION, atol=1e-6)
