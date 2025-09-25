# Example programs

## Online learning examples

 To compare the performance between the NumPy and Taichi (GPU) backends using the `run_online_learning_test.py` script, you can use the following command lines.

-----

### Running with the NumPy Backend

This command forces the script to use the pure NumPy implementation for all computations.

```bash
uv run python examples/run_online_learning_test.py experiment.use_numpy_version=true
```

  * `experiment.use_numpy_version=true`: This overrides the default configuration to select the `NumpyEchoStateNetwork` and its corresponding solvers.

-----

### Running with the Taichi Backend (GPU)

This command will use the Taichi implementation, leveraging the GPU for computation as specified by the default configuration.

```bash
uv run python examples/run_online_learning_test.py taichi.backend=gpu
```

  * `taichi.backend=gpu`: This explicitly sets the Taichi backend to use the GPU. The script will use the `EchoStateNetwork` and the Taichi versions of the LMS and RLS solvers.

-----

### Comparing Performance with Different Parameters

You can also override any other parameter from the command line to see how it affects performance. For example, to test with a larger reservoir size of 10000 for both backends, you would run:

**NumPy with a larger reservoir:**

```bash
uv run python examples/run_online_learning_test.py experiment.use_numpy_version=true esn.n_reservoir=10000
```

**Taichi (GPU) with a larger reservoir:**

```bash
uv run python examples/run_online_learning_test.py taichi.backend=gpu esn.n_reservoir=10000
```

By timing these commands, you can get a good sense of the performance difference between the two backends for your online learning task.
