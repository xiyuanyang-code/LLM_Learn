import torch.nn as nn
import torch
import torch.nn.functional as F
from d2l import torch as d2l
from torch.utils.data import Dataset, DataLoader


def show_heatmaps(
    matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap="Reds"
):
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]

    # it means the total numbers of images
    print(num_cols, num_rows)

    fig, axes = d2l.plt.subplots(
        num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            # print(type(matrix)): torch.Tensor
            # print(type(ax))
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    fig.savefig("img/heatmaps.png")


def plot_kernel_reg(y_hat, x_test, y_truth, x_train, y_train):
    d2l.plot(
        x_test,
        [y_truth, y_hat],
        "x",
        "y",
        legend=["Truth", "Pred"],
        xlim=[0, 10],
        ylim=[-1, 5],
    )
    d2l.plt.plot(x_train, y_train, "o", alpha=0.5)
    d2l.plt.savefig("my_img/kernel_reg.png")


def generate_datasets(width: float, n_train: int, n_test: int):
    """Generate random datasets for NWKernel regression."""
    x_train = torch.sort(torch.rand(n_train) * width).values

    def target_function(x):
        return 2 * torch.sin(x) + x**0.8

    y_train = target_function(x_train) + torch.normal(0.0, 0.5, (n_train,))
    x_test = torch.linspace(0, width, n_test)
    y_truth = target_function(x_test)

    print(f"Generated datasets - Train: {n_train}, Test: {n_test}")
    return x_train, y_train, x_test, y_truth


class NWKernelRegression(nn.Module):
    def __init__(self, x_train, y_train):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        # We use one weight per training example
        self.w = nn.Parameter(torch.ones(len(x_train)), requires_grad=True)

    def forward(self, queries):
        # queries: shape [n_queries]
        # Expand queries to compare with all training examples
        queries = queries.unsqueeze(1)  # [n_queries, 1]
        keys = self.x_train.unsqueeze(0)  # [1, n_train]

        # Compute attention weights
        diff = queries - keys  # [n_queries, n_train]
        self.attention_weights = F.softmax(-((diff * self.w) ** 2) / 2, dim=1)

        # Weighted average of training outputs
        return torch.matmul(self.attention_weights, self.y_train)


def visualize_attention(net, x_test, x_train, num_points=3):
    """Visualize attention weights for a few test points."""
    import matplotlib.pyplot as plt

    idxs = torch.linspace(0, len(x_test) - 1, num_points).long()
    queries = x_test[idxs]
    keys = x_train
    with torch.no_grad():
        for i, query in enumerate(queries):
            diff = query - keys
            attn = torch.softmax(-(diff * net.w).pow(2) / 2, dim=0)
            plt.figure()
            plt.title(f"Attention for test x={query.item():.2f}")
            plt.plot(keys, attn.numpy(), "o-")
            plt.xlabel("x_train")
            plt.ylabel("Attention weight")
            plt.savefig("my_img/attention.png")
            plt.close()


def visualize_kernel_shape(net, x_train):
    """Visualize the learned kernel shape centered at a point."""
    import matplotlib.pyplot as plt

    center = x_train[len(x_train) // 2]
    diffs = torch.linspace(-5, 5, 100)
    with torch.no_grad():
        attn = torch.softmax(-(diffs * net.w.mean()).pow(2) / 2, dim=0)
    plt.figure()
    plt.title("Learned Kernel Shape")
    plt.plot(diffs.numpy(), attn.numpy())
    plt.xlabel("x - center")
    plt.ylabel("Kernel value")
    plt.savefig("my_img/kernal.png")
    plt.close()


def visualize_training_process(record_epoch, record_loss):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("Training Process")
    plt.plot(record_epoch, record_loss)
    plt.xlabel("epoches")
    plt.ylabel("Training Loss")
    plt.savefig("my_img/loss.png")
    plt.close()


def train(width, epochs, n_train, n_test):
    # Generate data
    x_train, y_train, x_test, y_truth = generate_datasets(width, n_train, n_test)

    # Initialize model and optimizer
    net = NWKernelRegression(x_train, y_train)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    record_loss = []
    record_epochs = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = net(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        record_loss.append(loss.item())
        record_epochs.append(epoch)

        if epoch % 10 == 0:
            # animator.add(epoch, loss.item())
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    visualize_training_process(record_epochs, record_loss)

    # Testing
    with torch.no_grad():
        y_hat = net(x_test)
    plot_kernel_reg(y_hat, x_test, y_truth, x_train, y_train)
    visualize_attention(net, x_test, x_train)
    visualize_kernel_shape(net, x_train)

    show_heatmaps(
        net.attention_weights.unsqueeze(0).unsqueeze(0),
        xlabel="training inputs",
        ylabel="testing inputs",
    )


class SimpleTest:
    def test_data_generation(
        self, x_train, y_train, x_test, y_truth, n_train, n_test, width
    ):
        assert x_train.shape == (n_train,)
        assert y_train.shape == (n_train,)
        assert x_test.shape == (n_test,)
        assert y_truth.shape == (n_test,)
        assert torch.allclose(x_test[0], torch.tensor(0.0))
        assert torch.allclose(x_test[-1], torch.tensor(float(width)))
        print("✅ Data generation tests passed")


if __name__ == "__main__":
    # Parameters
    width = 10.0
    n_train = 100
    n_test = 100
    epochs = 1000

    # Run tests
    tester = SimpleTest()
    x_train, y_train, x_test, y_truth = generate_datasets(width, n_train, n_test)
    tester.test_data_generation(
        x_train, y_train, x_test, y_truth, n_train, n_test, width
    )

    # Train and evaluate
    train(width, epochs, n_train, n_test)
