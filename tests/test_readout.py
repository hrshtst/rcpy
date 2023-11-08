import numpy as np
import pytest
from rebasicspy.readout import Readout


@pytest.fixture
def readout() -> Readout:
    return Readout()


def dummy_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(shape=(15,))
    y_target = np.zeros(shape=(3,))
    return x, y_target


class TestReadout:
    def test_init(self, readout: Readout):
        assert readout.batch_interval == 0
        assert readout._batch_count == 0
        assert readout._batch_processed is False
        assert readout._batch_finalized is False

    def test_Wout_raise_exception(self, readout: Readout):
        with pytest.raises(RuntimeError) as exinfo:
            _ = readout.Wout
        assert str(exinfo.value).startswith("Optimization of readout layer has not started yet.")

    def test_backward_count_number(self, readout: Readout):
        assert readout._batch_count == 0
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout._batch_count == 1
        readout.backward(x, y_target)
        assert readout._batch_count == 2
        for expected in range(3, 10):
            readout.backward(x, y_target)
            assert readout._batch_count == expected

    def test_backward_make_processed_flag_off(self, readout: Readout):
        assert readout._batch_processed is False
        readout._batch_processed = True
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout._batch_processed is False
        assert readout._batch_finalized is False

    def test_process_backward_batch_reset_count_to_zero(self, readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        assert readout._batch_count == 10
        readout.process_backward_batch()
        assert readout._batch_count == 0

    def test_process_backward_batch_make_process_flag_on(self, readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        assert readout._batch_processed is False
        readout.process_backward_batch()
        assert readout._batch_processed is True
        assert readout._batch_finalized is False

    def test_finalize_backward_batch_reset_count_to_zero(self, readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        assert readout._batch_count == 10
        readout.finalize_backward_batch()
        assert readout._batch_count == 0

    def test_finalize_backward_batch_make_finalize_flag_on(self, readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        assert readout._batch_finalized is False
        readout.finalize_backward_batch()
        assert readout._batch_finalized is True

    def test_finalize_backward_batch_then_backward_make_finalize_flag_off(self, readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        readout.finalize_backward_batch()
        assert readout._batch_finalized is True
        readout.backward(x, y_target)
        assert readout._batch_finalized is False

    def test_backward_never_process_batch_when_interval_is_zero(self, readout: Readout):
        readout.batch_interval = 0
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
            assert readout._batch_processed is False

    def test_backward_run_batch_process_in_interval(self, readout: Readout):
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

    def test_reset(self, readout: Readout):
        x, y_target = dummy_data()
        for _ in range(10):
            readout.backward(x, y_target)
        readout.finalize_backward_batch()
        assert readout._batch_processed is True
        assert readout._batch_finalized is True
        readout.reset()
        assert readout._batch_processed is False
        assert readout._batch_finalized is False

    def test_reset_del_wout(self, readout: Readout):
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        readout.finalize_backward_batch()
        readout._Wout = np.zeros((len(x), len(x)))
        readout.reset()
        assert not hasattr(readout, "_Wout")

    def test_process_backward_batch_make_unprocessed_batch_false(self, readout: Readout):
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout.has_unprocessed_batch() is True
        readout.process_backward_batch()
        assert readout.has_unprocessed_batch() is False

    def test_reset_make_unprocessed_batch_false(self, readout: Readout):
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout.has_unprocessed_batch() is True
        readout.reset()
        assert readout.has_unprocessed_batch() is False

    def test_process_backward_batch_make_nothing_to_process_false(self, readout: Readout):
        assert readout.has_nothing_to_process() is True
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout.has_nothing_to_process() is False
        readout.process_backward_batch()
        assert readout.has_nothing_to_process() is False

    def test_reset_make_nothing_to_process_true(self, readout: Readout):
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout.has_nothing_to_process() is False
        readout.finalize_backward_batch()
        assert readout.has_nothing_to_process() is False
        readout.reset()
        assert readout.has_nothing_to_process() is True

    def test_fit_finalize_when_unprocessed_batch_remain(self, readout: Readout):
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        assert readout._batch_finalized is False
        try:
            readout.fit()
        except NotImplementedError:
            pass
        assert readout._batch_finalized is True

    def test_fit_raise_when_right_after_reset(self, readout: Readout):
        x, y_target = dummy_data()
        readout.backward(x, y_target)
        readout.reset()
        with pytest.raises(RuntimeError):
            readout.fit()
