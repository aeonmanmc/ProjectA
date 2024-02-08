from pathlib import Path

import torch
from matplotlib import pyplot as plt


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def preview_element(data, index):
    image, label = data[index]
    # print(f"Image shape: {image.shape}")
    class_names = data.classes
    plt.title(class_names[label])
    plt.imshow(image.squeeze())
    plt.show()
