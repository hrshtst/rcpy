import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from rebasicspy.metrics import mean_squared_error


@pytest.mark.parametrize(
    "y,y_target,expected",
    [
        ([2.5, 0.0, 2, 8], [2.5, 0.0, 2, 8], 0.0),
        ([2.5, 0.0, 2, 8], [3, -0.5, 2, 7], 0.375),
        ([2.5, 0.0, 2, 8], [3, -0.5, 2.5, 7.5], 0.25),
    ],
)
def test_mean_squared_error_calculate_correctly_with_one_dimension(
    y: list[float], y_target: list[float], expected: float
):
    assert mean_squared_error(y, y_target) == expected


@pytest.mark.parametrize(
    "y,y_target,expected",
    [
        ([2.5, 0.0, 2, 8], [2.5, 0.0, 2, 8], 0.0),
        ([2.5, 0.0, 2, 8], [3, -0.5, 2, 7], 0.6123724356957945),
        ([2.5, 0.0, 2, 8], [3, -0.5, 2.5, 7.5], 0.5),
    ],
)
def test_mean_squared_error_calculate_rooted_mse_with_one_dimension(
    y: list[float], y_target: list[float], expected: float
):
    assert mean_squared_error(y, y_target, root=True) == pytest.approx(expected)


@pytest.mark.parametrize(
    "y,y_target,expected",
    [
        ([2.5, 0.0, 2, 8], [2.5, 0.0, 2, 8], 0.0),
        ([2.5, 0.0, 2, 8], [3, -0.5, 2, 7], 0.375),
        ([2.5, 0.0, 2, 8], [3, -0.5, 2.5, 7.5], 0.25),
    ],
)
def test_mean_squared_error_calculate_correctly_with_one_dimensional_numpy_array(
    y: list[float], y_target: list[float], expected: float
):
    assert mean_squared_error(np.array(y), np.array(y_target)) == expected


@pytest.mark.parametrize(
    "y,y_target,expected",
    [
        ([2.5, 0.0, 2, 8], [2.5, 0.0, 2, 8], 0.0),
        ([2.5, 0.0, 2, 8], [3, -0.5, 2, 7], 0.6123724356957945),
        ([2.5, 0.0, 2, 8], [3, -0.5, 2.5, 7.5], 0.5),
    ],
)
def test_mean_squared_error_calculate_rooted_mse_with_one_dimensional_numpy_array(
    y: list[float], y_target: list[float], expected: float
):
    assert mean_squared_error(np.array(y), np.array(y_target), root=True) == pytest.approx(expected)


@pytest.mark.parametrize(
    "y,y_target,expected",
    [
        ([[0, 2], [-1, 2], [8, -5]], [[0.5, 1], [-1, 1], [7, -6]], 0.7083333333333333),
    ],
)
def test_mean_squared_error_calculate_with_two_dimensions(
    y: list[list[float]], y_target: list[list[float]], expected: float
):
    output: float = mean_squared_error(y, y_target)
    assert output == pytest.approx(expected)


@pytest.mark.parametrize(
    "y,y_target,expected",
    [
        ([[0, 2], [-1, 2], [8, -5]], [[0.5, 1], [-1, 1], [7, -6]], [0.416666666666667, 1.0]),
    ],
)
def test_mean_squared_error_calculate_raw_values_with_two_dimensions(
    y: list[list[float]], y_target: list[list[float]], expected: list[float]
):
    output: np.ndarray = mean_squared_error(y, y_target, raw_values=True)
    assert_almost_equal(output, expected)


@pytest.mark.parametrize(
    "y,y_target,expected",
    [
        ([[0, 2], [-1, 2], [8, -5]], [[0.5, 1], [-1, 1], [7, -6]], 0.8227486122),
    ],
)
def test_mean_squared_error_calculate_rooted_mse_with_two_dimensions(
    y: list[float], y_target: list[float], expected: float
):
    output: float = mean_squared_error(y, y_target, root=True)
    assert output == pytest.approx(expected)


@pytest.mark.parametrize(
    "y,y_target,expected",
    [
        ([[0, 2], [-1, 2], [8, -5]], [[0.5, 1], [-1, 1], [7, -6]], [0.6454972244, 1.0]),
    ],
)
def test_mean_squared_error_calculate_raw_values_of_rooted_mse_with_two_dimensions(
    y: list[float], y_target: list[float], expected: list[float]
):
    output: np.ndarray = mean_squared_error(y, y_target, root=True, raw_values=True)
    assert_almost_equal(output, expected)
