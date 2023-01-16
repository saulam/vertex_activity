import torch

# train Generator with Wasserstein Loss
def generator_train_step(batch_size, generator, optimizer, real_charge, real_signal,
                         labels, noise_size, loss_fn, device):
    generator.train()

    # init gradient
    optimizer.zero_grad()

    # labels
    real_labels = labels.to(device)

    # fake
    z = torch.distributions.uniform.Uniform(-1, 1).sample([batch_size, 5 * 5 * 5, noise_size]).to(device)
    fake_signal, fake_charge = generator(real_labels, z)
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
    z = torch.distributions.uniform.Uniform(-1, 1).sample([batch_size, 5 * 5 * 5, noise_size]).to(device)
    fake_signal, fake_charge = generator(real_labels, z)
    fake_signal = fake_signal.reshape(-1, 2)

    # Generator loss
    loss = loss_fn(fake_signal, real_signal, fake_charge, real_charge)

    return loss