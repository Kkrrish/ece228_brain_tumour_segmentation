import matplotlib.pyplot as plt
import numpy as np
import torch # type: ignore


def plot_mask_channels(mask):
    mask_labels = ['NEC', 'ED', 'ET']
    fig, axs = plt.subplots(1, 3, figsize=(9.75, 5))
    for i, ax in enumerate(axs):
        new_mask = np.zeros((mask.shape[1], mask.shape[2]))

        #new_mask = mask[i, :, :]
        new_mask = (mask[i, :, :] > 0.5*np.max(mask[i, :, :]))
        ax.imshow(new_mask)
        ax.axis('off')
        ax.set_title(mask_labels[i])
    plt.tight_layout()
    plt.show()

def plot_mask_channels2(mask, title='Mask Channels as RGB'):
    mask_labels = ['Necrotic (NEC)', 'Edema (ED)', 'Tumour (ET)']
    fig, axs = plt.subplots(1, 3, figsize=(9.75, 5))
    for i, ax in enumerate(axs):
        new_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        new_mask[..., i] = mask[i, :, :] * 255
        ax.imshow(new_mask)
        ax.axis('off')
        ax.set_title(mask_labels[i])
    plt.suptitle(title, fontsize=19, y=0.93)
    plt.tight_layout()
    plt.show()


def plot_mask_channels4(mask):
    mask_labels = ['NEC', 'ED', 'ET', 'NET']
    fig, axs = plt.subplots(1, 4, figsize=(9.75, 5))
    for i, ax in enumerate(axs):
        new_mask = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.uint8)
        new_mask = mask[i, :, :]
        ax.imshow(new_mask)
        ax.axis('off')
        ax.set_title(mask_labels[i])
    plt.tight_layout()
    plt.show()

def plot_image_channels(img):
    channel_labels = ['T1', 'T1C', 'T2', 'FLAIR']
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axs.flatten()):
        channel_img = img[i, :, :]
        ax.imshow(channel_img)
        ax.axis('off')
        ax.set_title(channel_labels[i])
    plt.tight_layout()
    plt.show()

def overlay_masks_on_image(image, mask):
    t1_slice = image[0, :, :]
    t1_normalized = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min())

    rgb_image = np.stack([t1_normalized] * 3, axis=-1)
    color_overlay = np.stack([mask[0, :, :], mask[1, :, :], mask[2, :, :]], axis=-1)
    rgb_image = np.where(color_overlay, color_overlay, rgb_image)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()


def plot_epoch_losses(train_losses, val_losses):
    plt.style.use('ggplot')
    plt.rcParams['text.color'] = '#333333'

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training and validation loss with an offset for epochs
    ax.plot([np.NaN] + train_losses, color='#636EFA', marker='o', linestyle='-', linewidth=2, markersize=5, label='Training Loss')
    ax.plot([np.NaN] + val_losses, color='#EFA363', marker='s', linestyle='-', linewidth=2, markersize=5, label='Validation Loss')

    # Adding title, labels, and formatting
    ax.set_title('Loss Over Epochs', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)

    ax.set_ylim(0, 1)
    
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def visualize_sample_predictions(model, sample_input, sample_target, device):
    sample_input, sample_target = sample_input.to(device), sample_target.to(device)

    # Get model predictions
    sample_pred = torch.sigmoid(model(sample_input))

    # Prepare image and masks for visualization
    img_array = sample_input.detach().cpu().numpy().squeeze(0)
    pred_mask = sample_pred.detach().cpu().numpy().squeeze(0)
    target_mask = sample_target.detach().cpu().numpy().squeeze(0)

    #print(np.max(pred_mask[0, :, :]))
    #print(np.max(pred_mask[1, :, :]))
    #print(np.max(pred_mask[2, :, :]))

    for i in range(pred_mask.shape[1]):
        for j in range(pred_mask.shape[2]):
            mx = np.max(pred_mask[:,i,j])
            for k in range(pred_mask.shape[0]):
                if(pred_mask[k, i, j] < 0.1):
                    pred_mask[k, i, j] = 0

    #pred_mask1 = pred_mask.transpose(1, 2, 0)
    #idx = np.argmax(pred_mask1, axis=2)
    #plt.imshow(idx)
    #plt.show()
    #print(idx.shape)
    #for i in range(pred_mask.shape[0]):
    #    pred_mask1[:, :, i] = (idx == (i+1))
    #pred_mask1 = pred_mask1.transpose(2, 0, 1)

    #print(pred_mask) 
    #print(pred_mask.shape)
    #print(pred_mask1.shape)
    #print(np.count_nonzero(idx == 0))
    #print(np.count_nonzero(idx == 1))
    #print(np.count_nonzero(idx == 2))
    # Display the input image, predicted mask, and target mask
    plot_image_channels(img_array)
    #print("Prediction")
    #plot_mask_channels(pred_mask)
    #print("Ground Truth")
    #plot_mask_channels(target_mask)
    print("Prediction")
    plot_mask_channels(pred_mask)
    print("Ground Truth")
    plot_mask_channels(target_mask)