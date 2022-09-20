import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed=0):
    # for reproductibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_conv_output_size(xlen, p, k, s=1):
    return (xlen + 2 * p - k) // s + 1


class MapsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][None, :], self.labels[index]


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(11 * 61, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, x):
        x = torch.flatten(x, 1)
        for fc in self.layers:
            x = F.relu(fc(x))
        return x


class ConvSimpleNet(nn.Module):
    def __init__(self, k1_size=(3, 3), n_conv_ch=32, pool_size=2, p_dropout=0):
        super(ConvSimpleNet, self).__init__()
        nrows_conv1 = get_conv_output_size(11, 0, k1_size[0])
        ncols_conv1 = get_conv_output_size(61, 0, k1_size[1])
        expected_size = (nrows_conv1 // pool_size) * (
            ncols_conv1 // pool_size
        )  # //2 for maxpooling
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_conv_ch, kernel_size=k1_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(expected_size * n_conv_ch, 64), nn.ReLU(), nn.Dropout(p=p_dropout)
        )
        self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p=p_dropout))
        self.fc3 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p=p_dropout))
        self.fc4 = nn.Sequential(
            nn.Linear(64, 3),
            nn.ReLU(),
        )
        self.fc_layers = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        for fc in self.fc_layers:
            x = fc(x)
        return x


class Conv1Net(nn.Module):
    def __init__(self, k1_size=(3, 3)):
        super(Conv1Net, self).__init__()

        nrows_conv1 = get_conv_output_size(11, 0, k1_size[0])
        ncols_conv1 = get_conv_output_size(61, 0, k1_size[1])
        expected_size = (nrows_conv1 // 2) * (ncols_conv1 // 2)

        # convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=k1_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # fully connected layers
        self.fc1 = nn.Sequential(nn.Linear(expected_size * 20, 64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(64, 3), nn.ReLU())
        self.fc_layers = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        for fc in self.fc_layers:
            x = fc(x)
        return x


class Conv2Net(nn.Module):
    def __init__(self, k1_size=(3, 3), k2_size=(3, 3)):
        super(Conv2Net, self).__init__()

        # computing sizes
        nrows_conv1 = get_conv_output_size(11, 0, k1_size[0])
        ncols_conv1 = get_conv_output_size(61, 0, k1_size[1])
        nrows_conv2 = get_conv_output_size(nrows_conv1 // 2, 0, k2_size[0])
        ncols_conv2 = get_conv_output_size(ncols_conv1 // 2, 0, k2_size[1])
        expected_size = (nrows_conv2 // 2) * (ncols_conv2 // 2)

        # layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=k1_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=k2_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Sequential(nn.Linear(20 * expected_size, 64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, 3, nn.ReLU()))
        self.fc_layers= [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = torch.flatten(x, 1)
        for fc in self.fc_layers:
            x = fc(x)
        return x


def train_validate(
    num_epochs,
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    lr_scheduler=None,
    validate=True,
):
    model.to(DEVICE)
    train_ep_loss, train_acc = np.zeros(num_epochs), np.zeros(num_epochs)
    val_ep_loss, val_acc = np.zeros(num_epochs), np.zeros(num_epochs)
    for ep in range(num_epochs):
        print(f"--- Epoch {ep + 1}/{num_epochs}: ")
        # training
        model.train()
        n_success = 0
        batch_losses = 0
        for i, (Xtr, ytr) in enumerate(train_dataloader):
            Xtr, ytr = Xtr.float().to(DEVICE), ytr.type(torch.LongTensor).to(DEVICE)
            # forward and loss
            y_pred = model(Xtr)
            loss = loss_fn(y_pred, ytr)
            # backprop and weigths updating
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # computing metrics
            pred_labels = torch.argmax(y_pred, dim=1)
            n_success += (torch.sum(pred_labels == ytr)).item()
            batch_losses += loss.item()
        # updating learning rate
        if lr_scheduler:
            lr_scheduler.step()
        # reporting metrics
        train_acc[ep] = n_success / len(train_dataloader.dataset)
        train_ep_loss[ep] = batch_losses / (i + 1)
        print(
            f"Train      -> loss = {train_ep_loss[ep]:.3f} | acc = {train_acc[ep]:.3f}"
        )

        # validation
        if validate:
            model.eval()
            n_success = 0
            batch_losses = 0
            with torch.no_grad():
                for i, (Xval, yval) in enumerate(val_dataloader):
                    Xval, yval = Xval.float().to(DEVICE), yval.type(
                        torch.LongTensor
                    ).to(DEVICE)
                    # forward and loss
                    y_pred = model(Xval)
                    loss = loss_fn(y_pred, yval)
                    # computing metrics
                    pred_labels = torch.argmax(y_pred, dim=1)
                    n_success += (torch.sum(pred_labels == yval)).item()
                    batch_losses += loss.item()
            val_acc[ep] = n_success / len(val_dataloader.dataset)
            val_ep_loss[ep] = batch_losses / (i + 1)
            print(
                f"Validation -> loss = {val_ep_loss[ep]:.3f} | acc = {val_acc[ep]:.3f}"
            )
    return train_ep_loss, val_ep_loss, train_acc, val_acc


def train_model(
    model,
    train_dataset,
    val_dataset,
    batch_size,
    num_epochs,
    lr,
    optimizer=None,
    scheduler=None,
):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_ep_loss, val_ep_loss, train_acc, val_acc = train_validate(
        num_epochs,
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        optimizer,
        lr_scheduler=scheduler,
        validate=True,
    )
    return train_ep_loss, val_ep_loss, train_acc, val_acc, model


def plot_train_val_loss(num_epochs, train_loss, train_acc, val_loss, val_acc, title=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(np.arange(num_epochs), train_loss, "-", color='darkgreen', label="train loss")
    plt.plot(np.arange(num_epochs), train_acc, "--", color='darkgreen', label="train acc")
    plt.plot(np.arange(num_epochs), val_loss, "r-", color='royalblue', label="validation loss")
    plt.plot(np.arange(num_epochs), val_acc, "r--", color='royalblue', label="validation acc")
    plt.legend()
    ax.grid()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    if title != "":
        ax.set_title(title)
    return fig
