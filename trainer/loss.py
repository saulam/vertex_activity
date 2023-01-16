import torch
import torch.nn as nn

class Combined2Losses(nn.Module):
    '''
    Combination of two input loss functions
    '''
    def __init__(self, loss_fn1, loss_fn2, alpha1=1.0, alpha2=1.0):
        super(Combined2Losses, self).__init__()
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, fake_signal, real_signal, fake_charge, real_charge):
        loss1 = self.loss_fn1(fake_signal, real_signal)
        loss2 = self.loss_fn2(fake_charge, real_charge)
        loss = self.alpha1*loss1 + self.alpha2*loss2
        return loss

class Combined3Losses(nn.Module):
    '''
    Combination of two input loss functions (the second one is also used
    to perform learn the total charge of each event)
    '''
    def __init__(self, loss_fn1, loss_fn2, alpha1=1.0, alpha2=1.0, alpha3=1.0, device=None):
        super(Combined3Losses, self).__init__()
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.device = device

    def forward(self, fake_signal, real_signal, fake_charge, real_charge):
        # mask charges
        mask = real_signal.bool()
        fake_charge2 = torch.where(mask.to(self.device), fake_charge, torch.tensor(0.).to(self.device))
        real_charge2 = torch.where(mask, real_charge, torch.tensor(0.))
        fake_total_charge = torch.sum(fake_charge2, dim=1).to(self.device)
        real_total_charge = torch.sum(real_charge2, dim=1).to(self.device)
        fake_charge = fake_charge[mask]
        real_charge = real_charge[mask]

        # real
        real_signal = real_signal.reshape(-1).to(self.device)
        real_charge = real_charge.to(self.device)

        # calc loss
        loss1 = self.loss_fn1(fake_signal, real_signal)
        loss2 = self.loss_fn2(fake_charge, real_charge)
        loss3 = self.loss_fn2(fake_total_charge, real_total_charge)
        loss = self.alpha1*loss1 + self.alpha2*loss2 + self.alpha3*loss3
        return loss

class Chi2Loss(nn.Module):
    def __init__(self):
        '''
        Chi2 Likelihood loss function
        '''
        super(Chi2Loss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-9):
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(y_true.shape[0], -1)
        loss = torch.mean(((y_pred - y_true) ** 2) / (y_pred + y_true + eps))
        return loss

class PoissonLikelihood_loss(nn.Module):
    def __init__(self):
        '''
        Poisson Likelihood loss function
        '''
        super(PoissonLikelihood_loss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-6):
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_true = y_true.view(y_true.shape[0], -1)

        """Custom loss function for Poisson model."""
        loss = torch.mean(y_pred - y_true * torch.log(y_pred + eps)) + torch.mean((charge_pred - charge_true) ** 2)
        return loss