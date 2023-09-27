import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.models.feature_extraction import create_feature_extractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(img, mean, std, title=None):
    """
    Display image for a normalized image tensor after reintroducing mean and std.
    """
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause as plots update

def visualize_model(model, dataloaders, class_names, mean, std, num_images=10):
    """
    Visualize some test-set images and their corresponding model predictions
    """
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    for i, (inputs, labels) in enumerate(dataloaders['test']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Make predictions
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Plot images with predictions
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title(f'predicted: {class_names[preds[j]]}')
            imshow(inputs.cpu().data[j], mean, std)

            if images_so_far == num_images:
                return 0
        return 0

def scatter_plot_classes(dataset, labels, class_labels, class_colors, legend, vector=None, vector_color=None):
    """
    Plot the scatter plot for a dataset with two classes, distinguishing by color
    Assumes 2D inputs.
    """
    # Extract class-wise data
    class0 = np.array([[x,y] for (x,y), lab in zip(dataset, labels) if lab==class_labels[0]])
    class1 = np.array([[x,y] for (x,y), lab in zip(dataset, labels) if lab==class_labels[1]])

    # Scatter plot the two classes
    plt.scatter(class0[:,0], class0[:,1], c=class_colors[0], label=legend[0])
    plt.scatter(class1[:,0], class1[:,1], c=class_colors[1], label=legend[1])
    plt.legend()

    # Plot a vector if specified
    if type(vector)==np.ndarray:
        plt.quiver(0, 0, vector[0], vector[1], color= vector_color)

    plt.show()

def get_reduced_activation_space_points(model, layers_out, data, num_samples=600):
    """
    Extract the model's intermediate output from layers specified in the layers_out list, given input data
    """
    # Define an activation extractor for the specified layers
    activation_extractors = create_feature_extractor(model, return_nodes={l:l for l in layers_out})
    activations = {l:[] for l in layers_out}

    # Iterate through the data, encoding the input to the activation spaces
    activation_labels = []
    cnt = 0
    for inputs, labels in data:
        inputs = inputs.to(device)
        batch_labels = labels
        batch_extraction = activation_extractors(inputs)
        for l in layers_out:
            activations[l].append(batch_extraction[l].flatten(start_dim=1).cpu().detach().numpy())
        activation_labels.append(batch_labels)
        
        # Update the number of datapoints seen, proceed until the required number of points are sampled
        cnt += inputs.shape[0]
        if cnt>=num_samples:
            break
        
    # Reduce each set of activation space vectors to two dimensions (separately for each layer output)
    pca = PCA(n_components=2)
    for l in layers_out:
        activations[l] = pca.fit_transform(np.concatenate(activations[l], axis=0)[:num_samples])

    # Combine labels to a single numpy array and return
    activation_labels = np.concatenate(activation_labels)[:num_samples]
    
    return activations, activation_labels


def get_sensitive_filenames(tcav, single_sample_train_dataloader, threshold_sensitivity=0.01):
    """
    Get the filenames of Concept images (acording to the tcav model for a given threshold sensitivity)
    """
    cnt=0
    arnie_images_filenames = []

    # Iterate through every sample in the train set
    for sample in tqdm(single_sample_train_dataloader):
        img = sample['data']
        label = sample['target']
        filename = sample['filename']

        # Compute Sensitivity for the sample
        sensitivity = tcav.get_sensitivity_on_input_example([(img, label)])[0]

        # Add the image filename to the arnie list if sensitivity is high
        if sensitivity>threshold_sensitivity:
            arnie_images_filenames.append(filename)
            cnt+=1

    print('Found ', cnt, ' Concept Images')
    return arnie_images_filenames