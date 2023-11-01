import re
import matplotlib.pyplot as plt

# Define the regular expression for extracting information from specific lines
line_pattern = r'E(\d+), VL: (\d+\.\d+), VA: (\d+\.\d+)'

# Create empty lists to store extracted information
epochs = []
batch_losses = []
batch_accuracies = []

# Read the training log file
log_file = "./HAM10000 UNET/loss_functions/iou/iou.log"  # Replace with the correct file path
with open(log_file, 'r') as file:
    log_lines = file.readlines()

# Iterate through each line in the log
for line in log_lines:
    # Use regular expressions to extract information
    match = re.search(line_pattern, line)
    
    if match:
        epoch_num, loss, accuracy = match.groups()
        epochs.append(int(epoch_num))
        batch_losses.append(float(loss))
        batch_accuracies.append(float(accuracy))

# Print the extracted information
for i in range(len(epochs)):
    print(f"Epoch: E{epochs[i]}, Loss: {batch_losses[i]}, Accuracy: {batch_accuracies[i]}")

# Create batch loss and batch accuracy plots
plt.figure(figsize=(12, 6))

# Batch Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, batch_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()

# Batch Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, batch_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
