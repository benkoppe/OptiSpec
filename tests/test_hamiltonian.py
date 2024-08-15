import pytest
import jax.numpy as jnp
from functools import partial

from optispec import hamiltonian as h


@pytest.fixture
def mode_filled_block():
    mode_frequencies = jnp.array([100, 1200])
    mode_couplings = jnp.array([0.7, 1.6])
    basis_set = (3, 3)

    # functions to get values for each mode
    m1 = partial(
        h._mode_offdiagonal_element,
        coupling=mode_couplings[0].astype(float),
        frequency=mode_frequencies[0].astype(float),
    )
    m2 = partial(
        h._mode_offdiagonal_element,
        coupling=mode_couplings[1].astype(float),
        frequency=mode_frequencies[1].astype(float),
    )

    expected_block = jnp.array(
        [
            [0.0, m2(0), 0.0, m1(0), 0.0, 0.0, 0.0, 0.0, 0.0],
            [m2(0), 0.0, m2(1), 0.0, m1(0), 0.0, 0.0, 0.0, 0.0],
            [0.0, m2(1), 0.0, 0.0, 0.0, m1(0), 0.0, 0.0, 0.0],
            [m1(0), 0.0, 0.0, 0.0, m2(0), 0.0, m1(1), 0.0, 0.0],
            [0.0, m1(0), 0.0, m2(0), 0.0, m2(1), 0.0, m1(1), 0.0],
            [0.0, 0.0, m1(0), 0.0, m2(1), 0.0, 0.0, 0.0, m1(1)],
            [0.0, 0.0, 0.0, m1(1), 0.0, 0.0, 0.0, m2(0), 0.0],
            [0.0, 0.0, 0.0, 0.0, m1(1), 0.0, m2(0), 0.0, m2(1)],
            [0.0, 0.0, 0.0, 0.0, 0.0, m1(1), 0.0, m2(1), 0.0],
        ]
    )

    return {
        "block": expected_block,
        "mode_frequencies": mode_frequencies,
        "mode_couplings": mode_couplings,
        "basis_set": basis_set,
    }


@pytest.fixture
def local_filled_block(mode_filled_block):
    state_energy = 0.0
    mode_frequencies = mode_filled_block["mode_frequencies"]
    mode_couplings = mode_filled_block["mode_couplings"]

    # function to get each diagonal value
    def d(m1_idx: int, m2_idx: int) -> float:
        m1_component = h._mode_diagonal_component(
            m1_idx,
            mode_frequencies[0],
            mode_couplings[0],
        )
        m2_component = h._mode_diagonal_component(
            m2_idx,
            mode_frequencies[1],
            mode_couplings[1],
        )
        return state_energy + m1_component + m2_component

    expected_diagonals = jnp.array(
        [
            d(0, 0),
            d(0, 1),
            d(0, 2),
            d(1, 0),
            d(1, 1),
            d(1, 2),
            d(2, 0),
            d(2, 1),
            d(2, 2),
        ]
    )

    expected_block = mode_filled_block["block"] + jnp.diag(expected_diagonals)

    return {
        **mode_filled_block,
        "state_energy": state_energy,
        "block": expected_block,
    }


def test_mode_block_creation(mode_filled_block):
    block = h._create_block_with_modes(
        True,
        mode_filled_block["basis_set"],
        (True, True),
        mode_filled_block["mode_frequencies"],
        mode_filled_block["mode_couplings"],
    )

    assert jnp.allclose(block, mode_filled_block["block"])


def test_local_block_creation(local_filled_block):
    params = h.Params(
        transfer_integrals=0,
        state_energies=jnp.array([local_filled_block["state_energy"]]),
        mode_basis_sets=local_filled_block["basis_set"],
        mode_localities=(True, True),
        mode_frequencies=local_filled_block["mode_frequencies"],
        mode_state_couplings=jnp.array([local_filled_block["mode_couplings"]]),
    )

    block = h._local_block(0, params)

    assert jnp.allclose(block, local_filled_block["block"])


def test_local_block_localities(local_filled_block):
    params = h.Params(
        transfer_integrals=0,
        state_energies=jnp.array([local_filled_block["state_energy"]]),
        mode_basis_sets=local_filled_block["basis_set"],
        mode_localities=(True, False),
        mode_frequencies=local_filled_block["mode_frequencies"],
        mode_state_couplings=jnp.array([local_filled_block["mode_couplings"]]),
    )

    block = h._local_block(0, params)

    expected_block = set_first_off_diagonal_to_zero(local_filled_block["block"])

    assert jnp.allclose(block, expected_block)


def test_non_local_block_creation(mode_filled_block):
    transfer_integral = 100.0

    params = h.Params(
        transfer_integrals=transfer_integral,
        state_energies=jnp.array([0]),
        mode_basis_sets=(3, 3),
        mode_localities=[False, False],
        mode_frequencies=mode_filled_block["mode_frequencies"],
        mode_state_couplings=jnp.array([mode_filled_block["mode_couplings"]]),
    )

    block = h._non_local_block(0, transfer_integral, params)

    expected_block = mode_filled_block["block"] + jnp.diag(
        jnp.repeat(transfer_integral, jnp.prod(jnp.array(params.mode_basis_sets)))
    )

    assert jnp.allclose(block, expected_block)


def test_non_local_block_creation_localities(mode_filled_block):
    transfer_integral = 100.0

    params = h.Params(
        transfer_integrals=transfer_integral,
        state_energies=jnp.array([0]),
        mode_basis_sets=(3, 3),
        mode_localities=[False, True],
        mode_frequencies=mode_filled_block["mode_frequencies"],
        mode_state_couplings=jnp.array([mode_filled_block["mode_couplings"]]),
    )

    block = h._non_local_block(0, transfer_integral, params)

    expected_block = mode_filled_block["block"] + jnp.diag(
        jnp.repeat(transfer_integral, jnp.prod(jnp.array(params.mode_basis_sets)))
    )

    expected_block = set_first_off_diagonal_to_zero(expected_block)

    assert jnp.allclose(block, expected_block)


def set_first_off_diagonal_to_zero(matrix):
    n = matrix.shape[0]
    mask = jnp.eye(n, k=1) + jnp.eye(n, k=-1)
    return jnp.where(mask == 1, 0.0, matrix)
