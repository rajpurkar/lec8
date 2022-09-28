import math
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_dataloader(is_train, batch_size, slice=5):
    full_dataset = torchvision.datasets.MNIST(
        root=".",
        train=is_train,
        transform=T.ToTensor(),
        download=True)
    sub_dataset = torch.utils.data.Subset(
        full_dataset,
        indices=range(0, len(full_dataset), slice))
    loader = torch.utils.data.DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=True,
        num_workers=2)
    return loader


def get_model(dropout):
    "A simple model"
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 10)).to(DEVICE)
    return model


def validate_model(
        model,
        valid_dl,
        loss_func,
        log_images=False,
        batch_idx=0):

    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            val_loss += loss_func(outputs, labels).item() * labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

class Config():
    """Helper to convert a dictionary to a class"""
    def __init__(
        self,
        dict):
        "A simple config class"
        self.epochs = dict['epochs']
        self.batch_size = dict['batch_size']
        self.lr = dict['lr']
        self.dropout = dict['dropout']


def train():
    # Launch 5 experiments, trying different dropout rates
    config_dict = {
        "epochs": 10,
        "batch_size": 128,
        "lr": 1e-3,
        "dropout": random.uniform(0.01, 0.80),
    }
    config = Config(config_dict)

    for _ in range(5):
        # Get the data
        train_dl = get_dataloader(
            is_train=True,
            batch_size=config.batch_size)
        valid_dl = get_dataloader(
            is_train=False,
            batch_size=2*config.batch_size)
        n_steps_per_epoch = \
            math.ceil(len(train_dl.dataset) / config.batch_size)
        
        # A simple MLP model
        model = get_model(config.dropout)

        # Make the loss and optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training
        example_ct = 0
        step_ct = 0
        for epoch in range(config.epochs):
            model.train()
            for step, (images, labels) in enumerate(train_dl):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                train_loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
        
                example_ct += len(images)
                step_ct += 1

            val_loss, accuracy = validate_model(
                model,
                valid_dl,
                loss_func,
                log_images=(epoch == (config.epochs-1)))

            print(f"Train Loss: {train_loss:.3f}, \
                Valid Loss: {val_loss:3f}, \
                Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    train()
