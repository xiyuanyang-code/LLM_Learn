import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def show_heatmaps(
    matrices,
    xlabel="Keys",
    ylabel="Queries",
    titles=None,
    figsize=(6, 6),
    cmap="Reds",
    save_path=None,
    show=True,
):
    """
    Display heatmaps for attention weights or other matrices.

    Args:
        matrices: Input tensor or array of shape (num_rows, num_cols, height, width)
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        titles: List of titles for subplots
        figsize: Size of the figure
        cmap: Color map
        save_path: Path to save the figure (None for not saving)
        show: Whether to display the figure
    """
    # Convert PyTorch tensor to numpy array if needed
    if hasattr(matrices, "detach"):
        matrices = matrices.detach().cpu().numpy()

    num_rows, num_cols = matrices.shape[0], matrices.shape[1]

    # Create figure and axes
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )

    # Plot each matrix
    for i in range(num_rows):
        for j in range(num_cols):
            pcm = axes[i, j].imshow(matrices[i, j], cmap=cmap)

            # Add labels only to the bottom row and leftmost column
            if i == num_rows - 1:
                axes[i, j].set_xlabel(xlabel)
            if j == 0:
                axes[i, j].set_ylabel(ylabel)

            # Add titles if provided
            if titles is not None:
                axes[i, j].set_title(titles[j])

    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to {save_path}")

    # Show plot if requested
    if show:
        plt.show()

    # Close the figure to prevent memory leaks
    plt.close()


def plot_kernel_reg(
    y_hat, x_test, y_truth, x_train, y_train, save_path="img/kernel_regression.png"
):
    """
    Plot kernel regression results and save the figure.

    Args:
        y_hat: Predicted values (tensor or array)
        x_test: Test input values
        y_truth: Ground truth values for test inputs
        x_train: Training input values
        y_train: Training target values
        save_path: Path to save the figure (default: "img/kernel_regression.png")
    """
    # Convert tensors to numpy if needed
    if hasattr(y_hat, "detach"):
        y_hat = y_hat.detach().cpu().numpy()
    if hasattr(y_truth, "detach"):
        y_truth = y_truth.detach().cpu().numpy()
    if hasattr(x_test, "detach"):
        x_test = x_test.detach().cpu().numpy()
    if hasattr(x_train, "detach"):
        x_train = x_train.detach().cpu().numpy()
    if hasattr(y_train, "detach"):
        y_train = y_train.detach().cpu().numpy()

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot truth and prediction lines
    plt.plot(x_test, y_truth, label="Truth")
    plt.plot(x_test, y_hat, label="Pred")

    # Plot training data points
    plt.scatter(x_train, y_train, marker="o", alpha=0.5, s=1, label="Training data")

    # Add labels and legend
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Kernel Regression Results")
    plt.legend()

    # Save figure with high quality
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_datasets(width: float, n_train: int, n_test: int):
    """Generate random datasets for NWKernel regression."""
    x_train = torch.sort(torch.rand(n_train) * width).values

    def target_function(x):
        return (
            2 * torch.sin(x) + 0.4 * torch.sin(3 * x) + 0.6 * torch.sin(6 * x) + x**0.5
        )

    y_train = target_function(x_train) + torch.normal(0.0, 0.5, (n_train,))
    x_test = torch.linspace(0, width, n_test)
    y_truth = target_function(x_test)

    print(f"Generated datasets - Train: {n_train}, Test: {n_test}")
    return x_train, y_train, x_test, y_truth


class NWKernelRegression(nn.Module):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.register_buffer("x_train", x_train)
        self.register_buffer("y_train", y_train)
        # We use one weight per training example
        self.w = nn.Parameter(torch.ones(len(x_train)), requires_grad=True)

    def forward(self, queries):
        queries = queries.unsqueeze(1)  # [n_queries, 1]
        keys = self.x_train.unsqueeze(0)  # [1, n_train]
        diff = queries - keys  # [n_queries, n_train]
        self.attention_weights = F.softmax(-((diff * self.w) ** 2) / 2, dim=1)
        return torch.matmul(self.attention_weights, self.y_train)


def visualize_attention(
    net, x_test, x_train, num_points=3, save_path="img/visualize_attention.png"
):
    """Visualize attention weights for a few test points."""

    idxs = torch.linspace(0, len(x_test) - 1, num_points).long()
    queries = x_test[idxs]
    keys = x_train
    w_cpu = net.w.detach().cpu()
    keys_cpu = keys.detach().cpu() if hasattr(keys, "detach") else torch.tensor(keys)
    with torch.no_grad():
        for i, query in enumerate(queries):
            query_cpu = (
                query.detach().cpu()
                if hasattr(query, "detach")
                else torch.tensor(query)
            )
            diff = query_cpu - keys_cpu
            attn = torch.softmax(-(diff * w_cpu).pow(2) / 2, dim=0)
            plt.figure()
            plt.title(f"Attention for test x={query_cpu.item():.2f}")
            plt.plot(keys_cpu, attn.numpy(), "o-")
            plt.xlabel("x_train")
            plt.ylabel("Attention weight")
            plt.savefig(save_path)
            plt.close()


def visualize_kernel_shape(net, x_train, save_path="img/kernelshape_visualize.png"):
    """Visualize the learned kernel shape centered at a point."""
    import matplotlib.pyplot as plt

    center = x_train[len(x_train) // 2]
    diffs = torch.linspace(-5, 5, 100)
    w_mean_cpu = net.w.mean().detach().cpu()
    with torch.no_grad():
        attn = torch.softmax(-(diffs * w_mean_cpu).pow(2) / 2, dim=0)
    plt.figure()
    plt.title("Learned Kernel Shape")
    plt.plot(diffs.numpy(), attn.numpy())
    plt.xlabel("x - center")
    plt.ylabel("Kernel value")
    plt.savefig(save_path)
    plt.close()


def visualize_training_process(
    record_epoch, record_loss, save_path="img/visualize_training_process.png"
):
    plt.figure()
    plt.title("Training Process")
    plt.plot(record_epoch, record_loss)
    plt.xlabel("epoches")
    plt.ylabel("Training Loss")
    plt.savefig(save_path)
    plt.close()


def train(width, epochs, n_train, n_test):
    # Generate data
    x_train, y_train, x_test, y_truth = generate_datasets(width, n_train, n_test)

    # Move data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_truth = y_truth.to(device)

    # Initialize model and optimizer
    net = NWKernelRegression(x_train, y_train).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
        net = nn.DataParallel(net)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    record_loss = []
    record_epochs = []

    plot_epochs = {10, 100, 1000, 5000, 10000, 20000, 50000}

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = net(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        record_loss.append(loss.item())
        record_epochs.append(epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        if (epoch + 1) in plot_epochs:
            with torch.no_grad():
                y_hat = (
                    net.module(x_test)
                    if isinstance(net, nn.DataParallel)
                    else net(x_test)
                )
            plot_kernel_reg(
                y_hat.cpu(),
                x_test.cpu(),
                y_truth.cpu(),
                x_train.cpu(),
                y_train.cpu(),
                save_path=f"img/kernel_regression_epoch{epoch+1}.png",
            )

    visualize_training_process(record_epochs, record_loss)

    # Testing
    with torch.no_grad():
        y_hat = net.module(x_test) if isinstance(net, nn.DataParallel) else net(x_test)

    model_for_vis = net.module if isinstance(net, nn.DataParallel) else net
    plot_kernel_reg(
        y_hat.cpu(), x_test.cpu(), y_truth.cpu(), x_train.cpu(), y_train.cpu()
    )
    visualize_attention(model_for_vis, x_test.cpu(), x_train.cpu())
    visualize_kernel_shape(model_for_vis, x_train.cpu())

    show_heatmaps(
        model_for_vis.attention_weights.unsqueeze(0).unsqueeze(0).cpu(),
        xlabel="training inputs",
        ylabel="testing inputs",
        titles="HeatMaps for the final attention",
    )


if __name__ == "__main__":
    # Parameters
    width = 100.0
    n_train = 10000
    n_test = 10000
    epochs = 60000

    # Run tests
    x_train, y_train, x_test, y_truth = generate_datasets(width, n_train, n_test)
    # Train and evaluate
    train(width, epochs, n_train, n_test)
