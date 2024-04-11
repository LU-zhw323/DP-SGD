import torch
import matplotlib.pyplot as plt
from torch import device, cuda
from torch.nn import CrossEntropyLoss
from torch.nn import Tanh, MaxPool2d, LogSoftmax, Flatten
from torchvision import datasets, transforms
import os
import numpy as np
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from torch import Tensor
from aihwkit.nn import AnalogLinear
from zhw323.DP.dp_sgd import DP_SGD
from zhw323.DP.dp_tile import DPRPUConfig, DPUpdateParameters
from zhw323.DP.dp_train import DPTrainer
from aihwkit.optim import AnalogOptimizer
model = AnalogLinear(2, 2)
model(Tensor([[0.1, 0.2], [0.3, 0.4]]))


def create_rpu_config():
    rpu_config = DPRPUConfig(update=DPUpdateParameters(batch_size=600, noise_multiplier=4.0, max_grad_norm=1.0))
    return rpu_config

def create_dp_trainer(model, rpu_config):
    dp_trainer = DPTrainer(model, rpu_config, train_set_size=60000,batch_size=600,noise_multiplier=4.0, max_grad_norm=1.0, lr=0.1,delta=1e-5)
    dp_trainer.create_DP_optimizer()
    return dp_trainer


def create_analog_network(rpu_config):
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(in_channels=1, out_channels=channel[0], kernel_size=5, stride=1,
                     rpu_config=rpu_config),
        Tanh(),
        MaxPool2d(kernel_size=2),
        AnalogConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1,
                     rpu_config=rpu_config),
        Tanh(),
        MaxPool2d(kernel_size=2),
        Tanh(),
        Flatten(),
        AnalogLinear(
            in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
        Tanh(),
        AnalogLinear(in_features=channel[3],
                     out_features=10, rpu_config=rpu_config),
        LogSoftmax(dim=1)
    )
    return model


criterion = CrossEntropyLoss()

DEVICE = device('cuda' if cuda.is_available() else 'cpu')
#DEVICE = device('cpu')
print('Running the simulation on: ', DEVICE)


def train_step(train_data, model, criterion, optimizer, trainer):
    """Train network.
    Args:
        train_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
    Returns:
        train_dataset_loss: epoch loss of the train dataset
    """
    total_loss = 0
    model.train()
    for images, labels in train_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        # Add training Tensor to the model (input).
        output = model(images)
        loss = criterion(output, labels)
        # Run training (backward propagation).
        loss.backward()
        # Optimize weights.
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    train_dataset_loss = total_loss / len(train_data.dataset)
    return train_dataset_loss


def test_step(validation_data, model, criterion):
    """Test trained network
    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
    Returns:
        test_dataset_loss: epoch loss of the train_dataset
        test_dataset_error: error of the test dataset
        test_dataset_accuracy: accuracy of the test dataset
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0
    model.eval()
    for images, labels in validation_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        pred = model(images)
        loss = criterion(pred, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        test_dataset_accuracy = predicted_ok/total_images*100
        test_dataset_error = (1-predicted_ok/total_images)*100
    test_dataset_loss = total_loss / len(validation_data.dataset)
    return test_dataset_loss, test_dataset_error, test_dataset_accuracy


def plot_results(train_losses, valid_losses, test_error):
    """Plot results.
    Args:
        train_losses (List): training losses as calculated in the training_loop
        valid_losses (List): validation losses as calculated in the training_loop
        test_error (List): test error as calculated in the training_loop
    """
    fig = plt.plot(train_losses, 'r-s', valid_losses, 'b-o')
    plt.title('aihwkit LeNet5')
    plt.legend(fig[:2], ['Training Losses', 'Validation Losses'])
    plt.xlabel('Epoch number')
    plt.ylabel('Loss [A.U.]')
    plt.grid(which='both', linestyle='--')
    plt.savefig(os.path.join(RESULTS, 'test_losses.png'))
    plt.close()
    fig = plt.plot(test_error, 'r-s')
    plt.title('aihwkit LeNet5')
    plt.legend(fig[:1], ['Validation Error'])
    plt.xlabel('Epoch number')
    plt.ylabel('Test Error [%]')
    plt.yscale('log')
    plt.ylim((5e-1, 1e2))
    plt.grid(which='both', linestyle='--')
    plt.savefig(os.path.join(RESULTS, 'test_error.png'))
    plt.close()


RESULTS = os.path.join(os.getcwd(), 'results', 'PCM-25-1')


def training_loop(model,trainer,criterion, train_data, validation_data, epochs=100, print_every=1):
    """Training loop.
    Args:
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        train_data (DataLoader): Validation set to perform the evaluation
        validation_data (DataLoader): Validation set to perform the evaluation
        epochs (int): global parameter to define epochs number
        print_every (int): defines how many times to print training progress
    """
    train_losses = []
    valid_losses = []
    test_error = []
    # Train model
    for epoch in range(0, epochs):
        # Train_step
        #epsilon = optimizer.privacyAccount(0.01, epoch+1, 1e-5)
        epsilon = trainer.optimizer.get_epsilon(epoch+1)
        train_loss = trainer.train_step(train_data,criterion)
        train_losses.append(train_loss)
        if epoch % print_every == (print_every - 1):
            # Validate_step
            with torch.no_grad():
                valid_loss, error, accuracy = test_step(
                    validation_data, model, criterion)
                valid_losses.append(valid_loss)
                test_error.append(error)
            print(f'Epoch: {epoch}\t'
                  f'Epsilon: {epsilon}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Test error: {error:.2f}%\t'
                  f'Test accuracy: {accuracy:.2f}%\t')
    # Save results and plot figures
    np.savetxt(os.path.join(RESULTS, "Test_error.csv"),
               test_error, delimiter=",")
    np.savetxt(os.path.join(RESULTS, "Train_Losses.csv"),
               train_losses, delimiter=",")
    np.savetxt(os.path.join(RESULTS, "Valid_Losses.csv"),
               valid_losses, delimiter=",")
    plot_results(train_losses, valid_losses, test_error)
    return model, trainer.optimizer, (train_losses, valid_losses, test_error)


def plot_results1(error_after_inference):
    fig = plt.plot(error_after_inference, 'r-s')
    plt.title('aihwkit LeNet5')
    plt.legend(fig[:1], ['Error after inference'])
    plt.xlabel('inference_time')
    plt.ylabel('Test Error [%]')
    plt.yscale('log')
    plt.ylim((5e-1, 1e2))
    plt.grid(which='both', linestyle='--')
    plt.savefig(os.path.join(RESULTS, 'Error after inference.png'))
    plt.close()


def test_inference(model, criterion, test_data):
    from numpy import logspace, log10
    error_after_inference = []
    total_loss = 0
    predicted_ok = 0
    total_images = 0
    accuracy_pre = 0
    error_pre = 0
    # Create the t_inference_list using inference_time.
    # Generate the 9 values between 0 and the inference time using log10
    max_inference_time = 1e6
    n_times = 9
    t_inference_list = [
        0.0] + logspace(0, log10(float(max_inference_time)), n_times).tolist()
    # Simulation of inference pass at different times after training.
    for t_inference in t_inference_list:
        model.drift_analog_weights(t_inference)
        time_since = t_inference
        accuracy_post = 0
        error_post = 0
        predicted_ok = 0
        total_images = 0
        for images, labels in test_data:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()
            accuracy_post = predicted_ok/total_images*100
            error_post = (1-predicted_ok/total_images)*100
        error_after_inference.append(error_post)
        print(f'Error after inference: {error_post:.2f}\t'
              f'Accuracy after inference: {accuracy_post:.2f}%\t'
              f'Drift t={time_since: .2e}\t')
        # Save results and plot figures
    np.savetxt(os.path.join(RESULTS, "Error after inference.csv"),
               error_after_inference, delimiter=",")
    plot_results1(error_after_inference)


PATH_DATASET = os.path.join('data', 'DATASET')
os.makedirs(PATH_DATASET, exist_ok=True)


def load_images():
    """Load images for train from torchvision datasets."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        PATH_DATASET, download=True, train=True, transform=transform)
    test_set = datasets.MNIST(
        PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(
        train_set, batch_size=600, shuffle=True)
    test_data = torch.utils.data.DataLoader(
        test_set, batch_size=600, shuffle=False)
    return train_data, test_data


torch.manual_seed(1)
# load the dataset
train_data, test_data = load_images()
# create the rpu_config
rpu_config = create_rpu_config()
# create the model
model = create_analog_network(rpu_config).to(DEVICE)
trainer = create_dp_trainer(model=model, rpu_config=rpu_config)
# define the analog optimizer
#optimizer = create_analog_optimizer(model)
"""optimizer = AnalogOptimizer(DP_SGD, model.parameters(
), lr=0.01, batch_size=600, noise_multiplier=4.0, max_grad_norm=1.0)
optimizer.createPrivacyAccount(total_size=60000, delta=1e-5, moment_order=range(2, 33))
optimizer.regroup_param_groups(model)"""
training_loop(model, trainer, criterion,train_data, test_data)
print('\nStarting Inference Pass:\n')
test_inference(model, criterion, test_data)


plt.plot(range(0, 100), trainer.optimizer.epsilon, label='delta=1e-5',
         color='blue')  # Plotting the data

# Adding title and labels
plt.title('PrivacyBudget')
plt.xlabel('Epoch')
plt.ylabel('Epislon')

# Optional: Adding legend if you have multiple lines
plt.legend()
plt.ylim(0, 5)

# Save the plot to a file
plt.savefig('./PrivacyLoss-LetNet5.png', dpi=300)
