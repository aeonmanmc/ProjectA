import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
from pathlib import Path
from torch import nn, Tensor
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
BATCH_SIZE = 32
MODEL_PATH = Path("models")
MODEL_NAME = "cv_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


def preview_element(data, index):
    image, label = data[index]
    # print(f"Image shape: {image.shape}")
    class_names = data.classes
    plt.title(class_names[label])
    plt.imshow(image.squeeze())
    plt.show()


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def save_model(model):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)


def make_prediction(model, element):
    model.eval()
    with torch.inference_mode():
        element = torch.unsqueeze(element, dim=0)
        pred_logit = model(element)
        pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
    return pred_prob


class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


def train_step(model: torch.nn.Module,
               data_loader: DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def main():
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model_0 = FashionMNISTModelV0(input_shape=784, hidden_units=10, output_shape=len(train_data.classes))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

    for epoch in tqdm(range(EPOCHS)):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_dataloader,
                   model=model_0,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn
                   )
    save_model(model_0)


def infer_model(data, index):
    loaded_model = FashionMNISTModelV0(input_shape=784, hidden_units=10, output_shape=10)
    loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    image_test, label_test = data[index]
    print(f'Actual: {data.classes[label_test]}')

    predicted_val = make_prediction(model=loaded_model, element=image_test)
    argmax_index = predicted_val.argmax()
    pred_label = data.classes[argmax_index]
    print(f'Prediction: {pred_label}')


# main()

infer_model(test_data, 27)
