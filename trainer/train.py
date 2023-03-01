import torch


def set_seed(seed_value, random, np, torch):
    """Set all the possible seeds to a constant value for reproducibility.

    Args:
        seed_value (int): The value to set the seeds to.
    """
    # Set the seed for the random number generator
    random.seed(seed_value)

    # Set the seed for numpy
    np.random.seed(seed_value)

    # Set the seed for PyTorch
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# train Generator with Wasserstein Loss
def generator_train_step(batch_size, generator, optimizer, real_charge, real_signal,
                         labels, noise_size, loss_fn, device):
    generator.train()

    # init gradient
    optimizer.zero_grad()

    # labels
    real_labels = labels.to(device)

    # fake
    z = torch.distributions.uniform.Uniform(-1, 1).sample([batch_size, 1, noise_size]).to(device)
    fake_signal, fake_charge = generator(None, real_labels, z)
    fake_signal = fake_signal.reshape(-1, 2)

    # Generator loss
    loss = loss_fn(fake_signal, real_signal, fake_charge, real_charge)

    loss.backward()
    optimizer.step()

    return loss


def generator_test_step(batch_size, generator, optimizer, real_charge, real_signal,
                        labels, noise_size, loss_fn, device):
    generator.eval()

    # init gradient
    optimizer.zero_grad()

    # labels
    real_labels = labels.to(device)

    # fake
    z = torch.distributions.uniform.Uniform(-1, 1).sample([batch_size, 1, noise_size]).to(device)
    fake_signal, fake_charge = generator(None, real_labels, z)
    fake_signal = fake_signal.reshape(-1, 2)

    # Generator loss
    loss = loss_fn(fake_signal, real_signal, fake_charge, real_charge)

    return loss