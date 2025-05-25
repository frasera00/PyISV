import torch
import torch.nn as nn
import torch.optim as optim

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def test_model_gradients():
    model = DummyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    input_data = torch.randn(10, 10)
    target_data = torch.randn(10, 10)

    for epoch in range(1, 6):
        optimizer.zero_grad()
        output = model(input_data)
        loss = nn.MSELoss()(output, target_data)
        loss.backward()
        
        grad_norms = {name: param.grad.norm().item() for name, param in model.named_parameters()}
        
        if epoch == 1:
            assert all(grad == 0 for grad in grad_norms.values()), "Gradients should be zero before backward pass"
        elif epoch > 1:
            assert all(grad > 0 for grad in grad_norms.values()), "Gradients should not be zero after backward pass"

test_model_gradients()