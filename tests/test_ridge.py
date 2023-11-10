import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from rebasicspy.ridge import Ridge

from ._base import RandDataSet, SmallDataSet, SmallDataSetMO


@pytest.fixture
def ridge() -> Ridge:
    return Ridge()


class TestRidge:
    def test_init(self):
        ridge = Ridge()
        assert ridge.regularization == 0
        assert hasattr(ridge, "_X") is False
        assert hasattr(ridge, "_Y") is False
        assert hasattr(ridge, "_XXT") is False
        assert hasattr(ridge, "_YXT") is False
        assert hasattr(ridge, "_sample_weight") is False

    @pytest.mark.parametrize("beta", [1e-3, 1, 1e10])
    def test_init_regularization(self, beta: float):
        ridge = Ridge(regularization=beta)
        assert ridge.regularization == beta

    def test_process_backward_check_stacked_matrix_size(self, ridge: Ridge):
        n_features = 4
        data = iter(RandDataSet(n_features=n_features))

        x, y = next(data)
        ridge.backward(x, y)
        assert ridge._X.shape == (1, n_features)
        assert ridge._Y.shape == (1, 1)

        x, y = next(data)
        ridge.backward(x, y)
        assert ridge._X.shape == (2, n_features)
        assert ridge._Y.shape == (2, 1)

        x, y = next(data)
        ridge.backward(x, y)
        assert ridge._X.shape == (3, n_features)
        assert ridge._Y.shape == (3, 1)

    def test_process_backward_make_stacked_matrix_float(self, ridge: Ridge):
        data = iter(RandDataSet(n_features=4))
        expected_type = np.dtype(float)

        x, y = next(data)
        ridge.backward(x, y)
        assert ridge._X.dtype is expected_type
        assert ridge._Y.dtype is expected_type

        x, y = next(data)
        ridge.backward(x, y)
        assert ridge._X.dtype is expected_type
        assert ridge._Y.dtype is expected_type

        x, y = next(data)
        ridge.backward(x, y)
        assert ridge._X.dtype is expected_type
        assert ridge._Y.dtype is expected_type

    def test_process_backward_check_stacked_matrix(self, ridge: Ridge):
        n_features = 4
        data = iter(RandDataSet(n_features=n_features))

        x, y = next(data)
        X = x.reshape(1, -1)
        Y = y.reshape(1, -1)
        ridge.backward(x, y)
        assert_array_equal(X, ridge._X)
        assert_array_equal(Y, ridge._Y)

        x, y = next(data)
        X = np.vstack((X, x.reshape(1, -1)))
        Y = np.vstack((Y, y.reshape(1, -1)))
        ridge.backward(x, y)
        assert_array_equal(X, ridge._X)
        assert_array_equal(Y, ridge._Y)

        x, y = next(data)
        X = np.vstack((X, x.reshape(1, -1)))
        Y = np.vstack((Y, y.reshape(1, -1)))
        ridge.backward(x, y)
        assert_array_equal(X, ridge._X)
        assert_array_equal(Y, ridge._Y)

    def test_process_backward_stack_vectors(self, ridge: Ridge):
        n_features = 4
        n_output = 2
        data = RandDataSet(n_features=n_features, n_output=n_output)
        for i, (x, y) in enumerate(data):
            ridge.backward(x, y)
            assert ridge._X.shape == (i + 1, n_features)
            assert ridge._Y.shape == (i + 1, n_output)

    def test_process_backward_batch(self, ridge: Ridge):
        n_features = 4
        data = iter(RandDataSet(n_features=n_features))
        ridge.batch_interval = 1

        x, y = next(data)
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        XXT = np.dot(x.T, x)
        YXT = np.dot(y.T, x)
        ridge.backward(x, y)
        assert_array_equal(XXT, ridge._XXT)
        assert_array_equal(YXT, ridge._YXT)

        x, y = next(data)
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        XXT = XXT + np.dot(x.T, x)
        YXT = YXT + np.dot(y.T, x)
        ridge.backward(x, y)
        assert_array_equal(XXT, ridge._XXT)
        assert_array_equal(YXT, ridge._YXT)

        x, y = next(data)
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        XXT = XXT + np.dot(x.T, x)
        YXT = YXT + np.dot(y.T, x)
        ridge.backward(x, y)
        assert_array_equal(XXT, ridge._XXT)
        assert_array_equal(YXT, ridge._YXT)

        x, y = next(data)
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        XXT = XXT + np.dot(x.T, x)
        YXT = YXT + np.dot(y.T, x)
        ridge.backward(x, y)
        assert_array_equal(XXT, ridge._XXT)
        assert_array_equal(YXT, ridge._YXT)

    @pytest.mark.parametrize("interval", [0, 1, 3, 10, 11])
    def test_fit_parametrize_batch_interval(self, ridge: Ridge, interval: int):
        n_samples = 10
        n_features = 4
        n_output = 2
        data = RandDataSet(n_samples=n_samples, n_features=n_features, n_output=n_output)
        X = np.empty((0, n_features), dtype=float)
        Y = np.empty((0, n_output), dtype=float)
        ridge.batch_interval = interval
        for x, y in data:
            X = np.vstack((X, x.reshape(1, -1)))
            Y = np.vstack((Y, y.reshape(1, -1)))
            ridge.backward(x, y)
        ridge.fit()
        assert hasattr(ridge, "_X") is False
        assert hasattr(ridge, "_Y") is False
        XXT = np.dot(X.T, X)
        YXT = np.dot(Y.T, X)
        assert_array_almost_equal(XXT, ridge._XXT)
        assert_array_almost_equal(YXT, ridge._YXT)

    def test_process_backward_stack_sample_weight(self, ridge: Ridge):
        n_samples = 10
        data = iter(RandDataSet(n_samples=n_samples))
        sample_weight = np.arange(n_samples)

        x, y = next(data)
        s = sample_weight[0]
        ridge.backward(x, y, s)
        assert_array_equal(sample_weight[:1], ridge._sample_weight)

        x, y = next(data)
        s = sample_weight[1]
        ridge.backward(x, y, s)
        assert_array_equal(sample_weight[:2], ridge._sample_weight)

        x, y = next(data)
        s = sample_weight[2]
        ridge.backward(x, y, s)
        assert_array_equal(sample_weight[:3], ridge._sample_weight)

    @pytest.mark.parametrize("solver", ["pseudoinv", "cholesky"])
    def test_fit_small_dataset(self, ridge: Ridge, solver: str):
        data = SmallDataSet()
        ridge.solver = solver
        for x, y in data:
            ridge.backward(x, y)
        ridge.fit()
        assert_array_almost_equal(data.coef, ridge.Wout)

    @pytest.mark.parametrize("solver", ["pseudoinv", "cholesky"])
    @pytest.mark.parametrize("beta", [0.00001, 0.01])
    def test_fit_small_dataset_with_regularization(self, ridge: Ridge, solver: str, beta: float):
        data = SmallDataSet()
        ridge.solver = solver
        ridge.regularization = beta
        for x, y in data:
            ridge.backward(x, y)
        ridge.fit()
        assert_array_almost_equal(data.coef, ridge.Wout, decimal=2)

    @pytest.mark.parametrize("solver", ["pseudoinv", "cholesky"])
    def test_fit_small_dataset_multi_output(self, ridge: Ridge, solver: str):
        data = SmallDataSetMO()
        ridge.solver = solver
        for x, y in data:
            ridge.backward(x, y)
        ridge.fit()
        assert_array_almost_equal(data.coef, ridge.Wout)
