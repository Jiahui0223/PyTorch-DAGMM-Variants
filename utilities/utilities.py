    
import torch
import torch.optim as optim
import matplotlib.pyplot as plt





def plot_loss(epoch_losses):
    """Plot the training loss per epoch."""
    # print(f"epoch_losses = {epoch_losses}")
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()