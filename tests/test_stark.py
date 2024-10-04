import pytest
import jax.numpy as jnp

from optispec.models import stark, two_state
from tests.data.stark import OLD_PACKAGE_ABSORPTION


@pytest.fixture
def stark_two_state_params():
    return stark.Params(
        model=two_state,
        neutral_params=two_state.Params(
            start_energy=0.0,
            end_energy=20_000.0,
            num_points=2_001,
            temperature_kelvin=300.0,
            broadening=200.0,
            coupling=100.0,
            energy_gap=8_000.0,
            mode_basis_sets=(20, 50),
            mode_frequencies=jnp.array([1200.0, 100.0]),
            mode_couplings=jnp.array([0.7, 2.0]),
        ),
        field_strength=0.01,
        positive_field_contribution_ratio=0.5,
        field_delta_dipole=38.0,
        field_delta_polarizability=1000.0,
    )


def test_stark_two_state_absorption_against_old_python_package(stark_two_state_params):
    params = stark_two_state_params
    absorption = stark.absorption(params)

    absorption = absorption.match_greatest_peak_of(OLD_PACKAGE_ABSORPTION.intensities)
    assert absorption.assert_similarity(OLD_PACKAGE_ABSORPTION)
