torch.manual_seed(13)

# Builds tensors from numpy arrays BEFORE split
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Builds dataset containing ALL data points
dataset = TensorDataset(x_tensor, y_tensor)

# Performs the split
train_data, val_data = random_split(dataset, [80, 20])

# Builds a loader of each set
train_loader = DataLoader(dataset=train_data, batch_size=16)
val_loader = DataLoader(dataset=val_data, batch_size=16)