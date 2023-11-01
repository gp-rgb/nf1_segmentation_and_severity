import re
import matplotlib.pyplot as plt
# Define regular expressions for extracting information
epoch_pattern = r'E(\d+),'
batch_pattern = r'BL: (\d+\.\d+), BA: (\d+\.\d+)'

# Create empty lists to store extracted information
epochs = []
batch_numbers = []
batch_losses = []
batch_accuracies = []

log_file = "./HAM10000 UNET/loss_functions/iou/iou.log"  # Replace with the correct file path
with open(log_file, 'r') as file:
    log_lines = file.readlines()

# Initialize variables to keep track of the current epoch and batch number
current_epoch = None
current_batch = 0

# Iterate through each line in the log
for line in log_lines:
    # Use regular expressions to extract information
    epoch_match = re.match(epoch_pattern, line)
    batch_match = re.search(batch_pattern, line)
    
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
    elif batch_match:
        current_batch += 1
        loss, accuracy = batch_match.groups()
        batch_numbers.append(current_batch)
        batch_losses.append(float(loss))
        batch_accuracies.append(float(accuracy))
        epochs.append(current_epoch)

# Print the extracted information
for i in range(len(epochs)):
    print(f"Epoch: E{epochs[i]}, Batch: {batch_numbers[i]}, Loss: {batch_losses[i]}, Accuracy: {batch_accuracies[i]}")

# Create loss and accuracy plots
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(batch_numbers, batch_losses, label='Batch Loss')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(batch_numbers, batch_accuracies, label='Batch Accuracy', color='orange')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()