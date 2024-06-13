import torch
import torch.nn as nn



def train_network(model, train_loader, val_loader, lr=0.001, epochs=20, device='cpu',steps_per_epoch=50, decay_rate=0.95, display_info=True):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_losses_per_epoch = []
    val_losses_per_epoch = []
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        # Adjust learning rate
        current_lr = lr * (decay_rate ** (epoch - 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Training phase
        model.train()
        total_train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
            if display_info: print(f"\rTraining batch: {batch_idx}/{steps_per_epoch}, Avg batch loss: {total_train_loss/batch_idx:.6f}", end='')

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
            total_train_loss += batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if batch_idx >= steps_per_epoch:
                if display_info: print()
                break
        train_losses_per_epoch.append(total_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader, start=1):
                if display_info: print(f"\rValidation batch: {batch_idx}/{steps_per_epoch}, Avg batch loss: {total_val_loss/batch_idx:.6f}", end='')

                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                total_val_loss += batch_loss.item()

                if batch_idx >= steps_per_epoch:
                    if display_info: print()
                    break
        val_losses_per_epoch.append(total_val_loss)

        if display_info: print(f"Epoch: {epoch}, Training loss: {total_train_loss:.6f}, Validation loss: {total_val_loss:.6f}, Learning rate: {current_lr:.6f}\n")
        
    print("Training finished.")
    return train_losses_per_epoch, val_losses_per_epoch

def save_model(model, path='saved_model'):
    torch.save(model, path)
