import torch
import torch.nn as nn
import torch.onnx
import onnx


# Create Simple Model
class Simple_Model(nn.Module):
    def __init__(self):
        super(Simple_Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = Simple_Model()
input_tensor = torch.randn(1, 28 * 28)
output = model(input_tensor)
print(output)

model_onnx = torch.onnx.export(
    model, input_tensor, "./onnx/simple_model.onnx", export_params=True
)
