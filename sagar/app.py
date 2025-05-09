import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Linear(10, 1).to(device)
x = torch.randn(100, 10).to(device)
y = torch.randn(100, 1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
