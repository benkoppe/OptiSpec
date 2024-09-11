import pytest

from optispec.models import mlj
from tests.data.mlj import OLD_PACKAGE_ABSORPTION


@pytest.fixture
def mlj_params():
    return mlj.Params()


def test_mlj_absorption_against_old_python_package(mlj_params):
    params = mlj_params
    absorption = mlj.absorption(params)

    absorption = absorption.match_greatest_peak_of(OLD_PACKAGE_ABSORPTION.intensities)
    assert absorption.assert_similarity(OLD_PACKAGE_ABSORPTION)
