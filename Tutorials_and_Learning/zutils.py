import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_2bit_flipflop_dataset(
    num_trials=100,
    trial_length=50,
    min_gap=1,
    max_gap=10,
    p_input=0.5,
    seed=None
):
    """
    Generate dataset for 2-bit flip-flop task.

    Args:
        num_trials (int): Number of trials to generate.
        trial_length (int): Number of time steps per trial.
        min_gap (int): Minimum time steps between inputs on a bit.
        max_gap (int): Maximum time steps between inputs on a bit.
        p_input (float): Probability of inserting a new input event at a candidate time step.
        seed (int or None): Random seed for reproducibility.

    Returns:
        inputs (torch.Tensor): Shape (num_trials, trial_length, 2)
        targets (torch.Tensor): Shape (num_trials, trial_length, 2)
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize data storage arrays
    inputs = np.zeros((num_trials, trial_length, 2), dtype=np.float64)
    targets = np.zeros((num_trials, trial_length, 2), dtype=np.float64)

    for trial in range(num_trials):
        last_values = np.array([0, 0], dtype=np.float64)
        next_input_time = np.random.randint(min_gap, max_gap + 1, size=2)

        for t in range(trial_length):
            for bit in range(2):
                if next_input_time[bit] == 0 and np.random.rand() < p_input:
                    val = np.random.choice([-1.0, 1.0])
                    inputs[trial, t, bit] = val
                    last_values[bit] = val
                    next_input_time[bit] = np.random.randint(min_gap, max_gap + 1)
                else:
                    next_input_time[bit] = max(0, next_input_time[bit] - 1)

            targets[trial, t] = last_values

    return torch.tensor(inputs), torch.tensor(targets)


import matplotlib.pyplot as plt

def plot_flipflop_trials(X, Y, num_trials_to_plot=5):
    """
    Plot input and target for the 2-bit flip-flop task.

    Args:
        X (Tensor): Input tensor of shape (num_trials, trial_length, 2)
        Y (Tensor): Target tensor of shape (num_trials, trial_length, 2)
        num_trials_to_plot (int): Number of trials to visualize.
    """
    num_trials = min(num_trials_to_plot, X.shape[0])
    trial_length = X.shape[1]
    time = np.arange(trial_length)

    fig, axs = plt.subplots(num_trials, 2, figsize=(12, 2 * num_trials), sharex=True)
    if num_trials == 1:
        axs = np.expand_dims(axs, 0)  # Handle single trial case

    for i in range(num_trials):
        # Inputs
        axs[i, 0].step(time, X[i, :, 0], where='mid', label="Input Bit 0", color='blue')
        axs[i, 0].step(time, X[i, :, 1], where='mid', label="Input Bit 1", color='red')
        axs[i, 0].set_ylim([-1.5, 1.5])
        axs[i, 0].set_ylabel(f'Trial {i}')
        axs[i, 0].set_title("Inputs")
        axs[i, 0].legend(loc='upper right')

        # Targets
        axs[i, 1].plot(time, Y[i, :, 0], label="Target Bit 0", color='blue', linestyle='--')
        axs[i, 1].plot(time, Y[i, :, 1], label="Target Bit 1", color='red', linestyle='--')
        axs[i, 1].set_ylim([-1.5, 1.5])
        axs[i, 1].set_title("Targets")
        axs[i, 1].legend(loc='upper right')

    axs[-1, 0].set_xlabel("Time Step")
    axs[-1, 1].set_xlabel("Time Step")
    plt.tight_layout()
    plt.show()


def flipflop_accuracy(predictions, targets):
    """
    Compute accuracy for the 2-bit flip-flop task.

    Args:
        predictions (Tensor): Output from the model, shape (batch, time, 2)
        targets (Tensor): True targets, shape (batch, time, 2)

    Returns:
        accuracy (float): Proportion of correct predictions.
    """
    # Round predictions to nearest of {-1, 0, 1}
    rounded_preds = torch.round(predictions)

    # Clip to valid range [-1, 1] in case outputs exceed it slightly
    rounded_preds = torch.clamp(rounded_preds, -1, 1)

    # Compare with targets
    correct = (rounded_preds == targets).all(dim=-1)  # both bits must be right
    accuracy = correct.float().mean().item()
    return accuracy

