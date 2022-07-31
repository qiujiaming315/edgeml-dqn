## Code

Various parts of the code are factored into different modules based on functionality.

- `utils.py`: Utility functions for printing messages, detecting a stop signal (`Ctrl+C`), and reading and writing checkpoints.
- `genseq.py`: Functions for generating synthetic sequences of image offloading metrics and inter-arrival times using different temporal models. Note that the offloading metric sequences are represented with image indices from the raw dataset to save memory.
- `data.py`: Code to load the data from the raw `npz` file, and a tuple sampler that randomly samples tuples of offloading metrics and inter-arrival times (for the current and the next state) and reward.
- `bucket.py`: Function to handle various bucket operations:
    - Convert rate and depth to integers.
    - Various utilities to interpret a long vector output from a neural network as a table of (n, a) values. Do operations to correctly do the whole `max_a' Q(n'=(n,a), a')`.
    - Code for computing an i.i.d. mdp policy.
- `model.py`: Functions to create different kinds of Keras models, including one from an iid policy threshold vector.
- `bstream.py`: Generates tensorflow compiled @tf.function's to sequentially simulate offloading decision making with a model and token bucket.
