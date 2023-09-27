import torch
import torchvision
from torchvision import datasets, models, transforms
import os
import shutil

# Define a custom dataset that returns filenames along with data and labels for samples
class DatasetWithFilename(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms= transforms
        self.dataset = datasets.ImageFolder(root=self.root, transform=self.transforms)
    def __getitem__(self, index):
        data, target = self.dataset[index]
        index = index
        return {
            "data": data,
            "target": target,
            "index": index,
            "folder": os.path.dirname(self.dataset.imgs[index][0]),
            "filename": os.path.basename(self.dataset.imgs[index][0]),
        }

    def __len__(self):
        return len(self.dataset)

def get_datasets_and_data_loaders(data_dir, mean, std, batch_size, split_dirs={'train':'train', 'test':'test'}):
    """
    Defines and returns the image_datasets, dataloaders, data_sizes and class_names
    for data found in the data_dir. Normalizes image tensors against the specified mean and std.
    """
    # Define data-transforms to be applied to images from the train and test splits
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # Define the image_datasets for the given datadir and specified transforms
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, split_dirs[x]),
                                              data_transforms[x])
                     for x in ['train', 'test']}

    # Create Dataloaders for training and evaluation
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4) for x in ['train', 'test']}

    # Create a single sample Dataset and Dataloader to iterate through the train set image-by-image along with filenames
    single_sample_train_dataset = DatasetWithFilename(os.path.join(data_dir, 'train'), data_transforms['train'])

    single_sample_train_dataloader = torch.utils.data.DataLoader(single_sample_train_dataset, batch_size=1,
                                                                 shuffle=False, num_workers=4)


    # Dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    # Class names
    class_names = image_datasets['train'].classes

    return (image_datasets, dataloaders, dataset_sizes, class_names, single_sample_train_dataloader)

def create_copy_of_train_set(copy_dir, data_dir):
    """
    Creates a copy of the train set directory files at the specified location
    Will first delete any previous version of a dataset that pre-exists at that location
    """
    # Original and correction dataset directory paths
    original_train_dataset_dir = os.path.join(data_dir,'train')

    # Delete the old corrected directory if one exists
    if os.path.exists(copy_dir):
        shutil.rmtree(copy_dir)

    # Copy imagefiles to a new corrected dataset directory
    shutil.copytree(original_train_dataset_dir, copy_dir)