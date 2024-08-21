# ruff: noqa: ERA001,PLR2004,SLF001
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from rebasicspy.lms import LMS

from ._base import LargeDataSet, SmallDataSet, SmallDataSetMO


@pytest.fixture
def lms() -> LMS:
    return LMS()


class TestLMS:
    def test_init(self) -> None:
        lms = LMS()
        assert lms.learning_rate == 0.1
        assert lms.batch_interval == 1
        assert lms._Wout_init == "zeros"

    @pytest.mark.parametrize("lr", [1.0, 0.01, 0.005])
    def test_init_learning_rate(self, lr: float) -> None:
        lms = LMS(lr)
        assert lms.learning_rate == lr

    def test_process_backward_initialize_Wout(self, lms: LMS) -> None:
        data = SmallDataSet()
        lms.learning_rate = 0
        x, y = next(iter(data))
        lms.backward(x, y)
        assert lms.Wout.shape == data.coef.shape
        assert_array_equal(lms.Wout, np.zeros(lms.Wout.shape))

    def test_process_backward_update_Wout_in_few_iterations(self, lms: LMS) -> None:
        data = iter(SmallDataSet())
        lms.learning_rate = 0.1
        # n = 0
        x, y = next(data)
        lms.backward(x, y)
        assert_array_almost_equal(np.array([[0.3, 0.3]]), lms.Wout)
        # n = 1
        x, y = next(data)
        lms.backward(x, y)
        assert_array_almost_equal(np.array([[0.71, 1.12]]), lms.Wout)
        # n = 2
        x, y = next(data)
        lms.backward(x, y)
        assert_array_almost_equal(np.array([[1.178, 1.588]]), lms.Wout)

    def test_process_backward_update_Wout_in_few_iterations_multi_output(self, lms: LMS) -> None:
        data = iter(SmallDataSetMO())
        lms.learning_rate = 0.1
        # n = 0
        x, y = next(data)
        lms.backward(x, y)
        assert_array_almost_equal(np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]), lms.Wout)
        # n = 1
        x, y = next(data)
        lms.backward(x, y)
        assert_array_almost_equal(np.array([[0.36, 0.62, 0.36], [0.22, 0.24, 0.22]]), lms.Wout)
        # n = 2
        x, y = next(data)
        lms.backward(x, y)
        assert_array_almost_equal(np.array([[0.224, 0.484, 0.224], [0.748, 0.768, 0.748]]), lms.Wout)

    def test_fit(self, lms: LMS) -> None:
        data = LargeDataSet()
        lms.learning_rate = 0.1
        for x, y in data:
            lms.backward(x, y)
        lms.fit()
        assert_array_almost_equal(data.coef, lms.Wout, decimal=1)


# Local Variables:
# jinx-local-words: "lr noqa"
# End:
