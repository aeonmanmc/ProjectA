import torch
import torchvision

import data_setup
import engine
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 3
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.1

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
data_transform = weights.transforms()


def main():
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=len(class_names),
                        bias=True)).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start training with help from engine.py
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           epochs=NUM_EPOCHS,
                           device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="pizza_model.pth")


if __name__ == '__main__':
    main()
