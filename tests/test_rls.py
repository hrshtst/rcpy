# ruff: noqa: ERA001,PLR2004,SLF001
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from rebasicspy.rls import RLS

from ._base import LargeDataSet, SmallDataSet, SmallDataSetMO


@pytest.fixture
def rls() -> RLS:
    return RLS()


class TestRLS:
    def test_init(self) -> None:
        rls = RLS()
        assert rls.forgetting_factor == 0.98
        assert rls.delta == 0.001
        assert rls.batch_interval == 1
        assert rls._Wout_init == "zeros"

    @pytest.mark.parametrize(("lr", "delta"), [(1.0, 0.01), (0.01, 0.00001), (0.005, 0.0)])
    def test_init_forgetting_factor(self, lr: float, delta: float) -> None:
        rls = RLS(lr, delta)
        assert rls.forgetting_factor == lr
        assert rls.delta == delta

    def test_process_backward_initialize_Wout_and_P(self, rls: RLS) -> None:
        data = SmallDataSet()
        rls.forgetting_factor = 0.98
        x, y = next(iter(data))
        rls.backward(x, y)
        assert rls.Wout.shape == data.coef.shape
        assert rls._P.shape == (len(x), len(x))

    def test_process_backward_update_Wout_in_few_iterations(self, rls: RLS) -> None:
        data = iter(SmallDataSet())
        rls.forgetting_factor = 0.1
        rls.delta = 0.001
        # n = 1
        x, y = next(data)
        rls.backward(x, y)
        w = 3000 / 2000.1
        assert_array_almost_equal(np.array([[w, w]]), rls.Wout)
        p = 1e6 / 2000.1
        assert_array_almost_equal(10 * np.array([[1e3 - p, -p], [-p, 1e3 - p]]), rls._P)

    def test_process_backward_update_Wout_in_few_iterations_multi_output(self, rls: RLS) -> None:
        data = iter(SmallDataSetMO())
        rls.forgetting_factor = 0.1
        rls.delta = 0.001
        # n = 1
        x, y = next(data)
        rls.backward(x, y)
        w1 = 1e3 / 3000.1
        w2 = 2e3 / 3000.1
        assert_array_almost_equal(np.array([[w1, w1, w1], [w2, w2, w2]]), rls.Wout)
        p = 1e6 / 3000.1
        expected_P = 10 * np.array([[1e3 - p, -p, -p], [-p, 1e3 - p, -p], [-p, -p, 1e3 - p]])
        assert_array_almost_equal(expected_P, rls._P)

    def test_fit(self, rls: RLS) -> None:
        data = LargeDataSet()
        rls.forgetting_factor = 0.98
        rls.delta = 1e-3
        for x, y in data:
            rls.backward(x, y)
        rls.fit()
        assert_array_almost_equal(data.coef, rls.Wout, decimal=1)
