def evaluate_model(model, test_data, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_data:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    average_loss = total_loss / len(test_data)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy