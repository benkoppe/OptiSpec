import pytest
import jax.numpy as jnp

from optispec import hamiltonian as h
from tests.data import (
    OLD_PACKAGE_MATRIX,
    OLD_PACKAGE_EIGENVALUES,
    OLD_PACKAGE_EIGENVECTORS,
)

# functions for testing the full matrix building and diagonalization
# compares blocks against old package results stored in a file (see fixtures below)

# TODO: extend to test against fortran results


@pytest.fixture
def old_package_comparison_params():
    return h.Params(
        transfer_integrals=100.0,
        state_energies=jnp.array([0.0, 8_000.0]),
        mode_basis_sets=(9, 10),
        mode_localities=(True, True),
        mode_frequencies=jnp.array([1200.0, 100.0]),
        mode_state_couplings=jnp.array([[0.0, 0.0], [0.7, 2.0]]),
    )


def test_matrix_building(old_package_comparison_params):
    params = old_package_comparison_params
    matrix = h.hamiltonian(params)

    assert jnp.allclose(matrix, OLD_PACKAGE_MATRIX)


def test_eigenvalues(old_package_comparison_params):
    params = old_package_comparison_params
    eigvals, _ = h.diagonalize(params)

    assert jnp.allclose(jnp.sort(eigvals), jnp.sort(OLD_PACKAGE_EIGENVALUES))
    # TODO: test eigenvectors
