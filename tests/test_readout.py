# ruff: noqa: PLR2004,SLF001
from __future__ import annotations

import contextlib

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from rcpy.readout import Readout, compute_error, initialize_weights, rescale_data


@pytest.fixture
def readout() -> Readout:
    return Readout()


def dummy_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(shape=(15,))
    y_target = np.zeros(shape=(3,))
    return x, y_target


class TestReadout:
    def test_init(self, readout: Readout) -> None:
        assert readout.batch_interval == 0
        assert readout._batch_count == 0
        assert readout._batch_processed is False
        assert readout._batch_finalized is False

    def test_Wout_raise_exception(self, readout: Readout) -> None:
        with pytest.raises(RuntimeError) as exinfo:
            _ = readout.Wout
        assert str(exinfo.value).startswith("Optimization of readout layer has not started yet.")

    def test_backward_count_number(self, readout: Readout) -> None:
        assert readout._batch_count == 0
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout._batch_count == 1
        readout.backward(x, y_target)
        assert readout._batch_count == 2
        for expected in range(3, 10):
            readout.backward(x, y_target)
            assert readout._batch_count == expected

    def test_backward_make_processed_flag_off(self, readout: Readout) -> None:
        assert readout._batch_processed is False
        readout._batch_processed = True
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout._batch_processed is False
        assert readout._batch_finalized is False

    def test_process_backward_batch_reset_count_to_zero(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        assert readout._batch_count == 10
        readout.process_backward_batch()
        assert readout._batch_count == 0

    def test_process_backward_batch_make_process_flag_on(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        assert readout._batch_processed is False
        readout.process_backward_batch()
        assert readout._batch_processed is True
        assert readout._batch_finalized is False

    def test_finalize_backward_batch_reset_count_to_zero(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        assert readout._batch_count == 10
        readout.finalize_backward_batch()
        assert readout._batch_count == 0

    def test_finalize_backward_batch_make_finalize_flag_on(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        assert readout._batch_finalized is False
        readout.finalize_backward_batch()
        assert readout._batch_finalized is True

    def test_finalize_backward_batch_then_backward_make_finalize_flag_off(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        readout.finalize_backward_batch()
        assert readout._batch_finalized is True
        readout.backward(x, y_target)
        assert readout._batch_finalized is False

    def test_backward_never_process_batch_when_interval_is_zero(self, readout: Readout) -> None:
        readout.batch_interval = 0
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
            assert readout._batch_processed is False

    def test_backward_run_batch_process_in_interval(self, readout: Readout) -> None:
        readout.batch_interval = 5
        x, y_target = dummy_data()
        # first interval
        for _ in range(4):
            readout.backward(x, y_target)
            assert readout._batch_processed is False
        readout.backward(x, y_target)
        assert readout._batch_processed is True
        # second interval
        for _ in range(4):
            readout.backward(x, y_target)
            assert readout._batch_processed is False
        readout.backward(x, y_target)
        assert readout._batch_processed is True

    def test_reset(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        readout.finalize_backward_batch()
        assert readout._batch_processed is True
        assert readout._batch_finalized is True
        readout.reset()
        assert readout._batch_processed is False
        assert readout._batch_finalized is False

    def test_reset_del_wout(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        readout.finalize_backward_batch()
        readout._Wout = np.zeros((len(x), len(x)))
        readout.reset()
        assert not hasattr(readout, "_Wout")

    def test_process_backward_batch_make_unprocessed_batch_false(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout.has_unprocessed_batch() is True
        readout.process_backward_batch()
        assert readout.has_unprocessed_batch() is False

    def test_reset_make_unprocessed_batch_false(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout.has_unprocessed_batch() is True
        readout.reset()
        assert readout.has_unprocessed_batch() is False

    def test_process_backward_batch_make_nothing_to_process_false(self, readout: Readout) -> None:
        assert readout.has_nothing_to_process() is True
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout.has_nothing_to_process() is False
        readout.process_backward_batch()
        assert readout.has_nothing_to_process() is False

    def test_reset_make_nothing_to_process_true(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout.has_nothing_to_process() is False
        readout.finalize_backward_batch()
        assert readout.has_nothing_to_process() is False
        readout.reset()
        assert readout.has_nothing_to_process() is True

    def test_fit_finalize_when_unprocessed_batch_remain(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout._batch_finalized is False
        with contextlib.suppress(NotImplementedError):
            readout.fit()
        assert readout._batch_finalized is True

    def test_fit_raise_when_right_after_reset(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        readout.reset()
        with pytest.raises(RuntimeError):
            readout.fit()

    def test_predict_raise_before_run_backward(self, readout: Readout) -> None:
        x, _ = dummy_data()
        with pytest.raises(RuntimeError):
            readout.predict(x)

    def test_predict_raise_if_right_after_reset(self, readout: Readout) -> None:
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        readout.reset()
        with pytest.raises(RuntimeError):
            readout.predict(x)


def get_data_for_rescale(
    n_samples: int = 10,
    n_features: int = 4,
    n_output: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.arange(n_samples * n_features).reshape(n_samples, n_features)
    y = np.arange(n_samples * n_output).reshape(n_samples, n_output)
    sw = np.arange(n_samples) + 1.0
    sw = 1.0 / sw
    sw_sqrt = np.sqrt(sw)
    expected_X = X * sw_sqrt[:, np.newaxis]
    expected_y = y * sw_sqrt[:, np.newaxis]
    if n_output == 1:
        y = np.ravel(y)
        expected_y = np.ravel(expected_y)
    return X, y, sw, expected_X, expected_y


@pytest.mark.parametrize("n_output", [2, 1])
def test_rescale_data(n_output: int) -> None:
    X, y, sw, expected_X, expected_y = get_data_for_rescale(n_output=n_output)
    actual_X, actual_y = rescale_data(X, y, sw)
    assert_array_equal(expected_X, actual_X)
    assert_array_equal(expected_y, actual_y)


def test_rescale_data_scalar() -> None:
    X, y, sw = np.array([1.0, 2.0]), 1.0, 4.0
    expected_X, expected_y = np.array([2.0, 4.0]), np.array([2.0])
    actual_X, actual_y = rescale_data(X, y, sw)
    assert_array_equal(expected_X, actual_X)
    assert_array_equal(expected_y, actual_y)


@pytest.mark.parametrize("shape", [(10,), (20, 4), (30, 1)])
def test_initialize_weights_random(shape: tuple[int, ...]) -> None:
    w = initialize_weights(shape, "random")
    assert w.shape == shape


@pytest.mark.parametrize("shape", [(10,), (20, 4), (30, 1)])
def test_initialize_weights_zeros(shape: tuple[int, ...]) -> None:
    w = initialize_weights(shape, "zeros")
    assert w.shape == shape
    for v in w.flatten():
        assert v == 0.0


def test_compute_error_scalar() -> None:
    r = Readout()
    r._Wout = np.array([[1.0, 2.0]])
    x = np.array([3.0, 5.0])
    y = 13.0
    e = compute_error(r, x, y)
    assert_array_equal(np.array([0.0]), e)


def test_compute_error_vector() -> None:
    r = Readout()
    r._Wout = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 2.0]])
    x = np.array([3.0, 5.0, 1.0])
    y = np.array([16.0, 31.0])
    e = compute_error(r, x, y)
    assert_array_equal(np.array([0.0, 0.0]), e)


# Local Variables:
# jinx-local-words: "Wout noqa"
# End:
