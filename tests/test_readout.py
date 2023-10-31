import numpy as np
import pytest
from rebasicspy.readout import Readout


@pytest.fixture
def default_readout() -> Readout:
    return Readout()


def dummy_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(shape=(15,))
    y_target = np.zeros(shape=(3,))
    return x, y_target


class TestReadout:
    def test_init(self, default_readout: Readout):
        assert default_readout.batch_interval == 0
        assert default_readout._batch_count == 0
        assert default_readout._batch_processed is False
        assert default_readout._batch_finalized is False

    def test_Wout_raise_exception(self, default_readout: Readout):
        with pytest.raises(RuntimeError) as exinfo:
            _ = default_readout.Wout
        assert str(exinfo.value).startswith("Optimization of readout layer has not started yet.")

    def test_backward_count_number(self, default_readout: Readout):
        assert default_readout._batch_count == 0
        x, y_target = dummy_data()
        default_readout.backward(x, y_target)
        assert default_readout._batch_count == 1
        default_readout.backward(x, y_target)
        assert default_readout._batch_count == 2
        for expected in range(3, 10):
            default_readout.backward(x, y_target)
            assert default_readout._batch_count == expected

    def test_backward_make_processed_flag_off(self, default_readout: Readout):
        assert default_readout._batch_processed is False
        default_readout._batch_processed = True
        x, y_target = dummy_data()
        default_readout.backward(x, y_target)
        assert default_readout._batch_processed is False
        assert default_readout._batch_finalized is False

    def test_process_backward_batch_reset_count_to_zero(self, default_readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            default_readout.backward(x, y_target)
        assert default_readout._batch_count == 10
        default_readout.process_backward_batch()
        assert default_readout._batch_count == 0

    def test_process_backward_batch_make_process_flag_on(self, default_readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            default_readout.backward(x, y_target)
        assert default_readout._batch_processed is False
        default_readout.process_backward_batch()
        assert default_readout._batch_processed is True
        assert default_readout._batch_finalized is False

    def test_finalize_backward_batch_reset_count_to_zero(self, default_readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            default_readout.backward(x, y_target)
        assert default_readout._batch_count == 10
        default_readout.finalize_backward_batch()
        assert default_readout._batch_count == 0

    def test_finalize_backward_batch_make_finalize_flag_on(self, default_readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            default_readout.backward(x, y_target)
        assert default_readout._batch_finalized is False
        default_readout.finalize_backward_batch()
        assert default_readout._batch_finalized is True

    def test_finalize_backward_batch_then_backward_make_finalize_flag_off(self, default_readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            default_readout.backward(x, y_target)
        default_readout.finalize_backward_batch()
        assert default_readout._batch_finalized is True
        default_readout.backward(x, y_target)
        assert default_readout._batch_finalized is False

    def test_backward_never_process_batch_when_interval_is_zero(self, default_readout: Readout):
        default_readout.batch_interval = 0
        x, y_target = dummy_data()
        for _ in range(10):
            default_readout.backward(x, y_target)
            assert default_readout._batch_processed is False

    def test_backward_run_batch_process_in_interval(self, default_readout: Readout):
        default_readout.batch_interval = 5
        x, y_target = dummy_data()
        # first interval
        for _ in range(4):
            default_readout.backward(x, y_target)
            assert default_readout._batch_processed is False
        default_readout.backward(x, y_target)
        assert default_readout._batch_processed is True
        # second interval
        for _ in range(4):
            default_readout.backward(x, y_target)
            assert default_readout._batch_processed is False
        default_readout.backward(x, y_target)
        assert default_readout._batch_processed is True

    def test_reset(self, default_readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            default_readout.backward(x, y_target)
        default_readout.finalize_backward_batch()
        assert default_readout._batch_processed is True
        assert default_readout._batch_finalized is True
        default_readout.reset()
        assert default_readout._batch_processed is False
        assert default_readout._batch_finalized is False

    def test_reset_always_finalize_when_unprocessed_yet(self, default_readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            default_readout.backward(x, y_target)
        assert default_readout._batch_count > 0
        assert default_readout._batch_finalized is False
        default_readout.reset()
        assert default_readout._batch_count == 0
        assert default_readout._batch_finalized is False  # flag gets to off
