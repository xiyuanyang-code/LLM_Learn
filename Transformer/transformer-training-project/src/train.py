import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import DataLoader
from src.model import TransformerModel

def train_model(epochs, learning_rate):
    # Initialize data loader and load data
    data_loader = DataLoader()
    train_data, val_data = data_loader.load_data()

    # Initialize the model
    model = TransformerModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_data:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'transformer_model.pth')

if __name__ == "__main__":
    train_model(epochs=10, learning_rate=0.001)