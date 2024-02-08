from pathlib import Path

import torch
import torchvision
from torchvision import datasets, transforms

MODEL_PATH = Path("models")
MODEL_NAME = "pizza_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


def make_prediction(model, element):
    model.eval()
    with torch.inference_mode():
        pred_logit = model(element)
        pred_prob = torch.softmax(pred_logit, dim=1)
    return pred_prob


def infer_model(img, class_names):
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    loaded_model = torchvision.models.efficientnet_b0(weights=weights)
    loaded_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=len(class_names),
                        bias=True))
    loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    image_transform = weights.transforms()
    transformed_image = image_transform(img).unsqueeze(dim=0)

    predicted_val = make_prediction(model=loaded_model, element=transformed_image)
    return predicted_val.argmax()


test_dir = "data/pizza_steak_sushi/test"
test_data = datasets.ImageFolder(test_dir)
print(f'Length: {len(test_data)}')
image_test, label_test = test_data[69]
print(f'Actual: {test_data.classes[label_test]}')

argmax_index = infer_model(image_test, test_data.classes)
pred_label = test_data.classes[argmax_index]
print(f'Prediction: {pred_label}')
