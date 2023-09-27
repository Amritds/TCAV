import random
import numpy as np
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from utils import scatter_plot_classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TCAV:
    def __init__(self, model, concept_examples, layer_out, class_index, random_dataset, negative_examples=None, use_known_negative_examples=False):
        """
        model: The network model used for class-prediction
        concept_examples: Images tensors of examples of the chosen concept
        layer_out: The name of the layer whose output corresponds to the chosen activation space
        class_index: The class-index (k) corresponding to the class_input_samples that will be used
        random_dataset: The dataset used to sample random example images (to train the cav classifier betweeen random and concept examples)
        """
        # Initialize object state variables
        self.model = model
        self.concept_examples = concept_examples
        self.num_samples = concept_examples.shape[0]
        self.layer_out = layer_out
        self.class_index = class_index
        self.random_dataset = random_dataset

        self.activation_extractor = self.get_activation_extractor()
        self.remainder_network = self.get_remainder_network()

        self.cav = None
        self.partial_gradients = None

        self.use_known_negative_examples = use_known_negative_examples
        self.negative_examples = negative_examples

        if negative_examples==None:
            # Define a random dataloader to sample random-image-samples
            self.random_dataloader = torch.utils.data.DataLoader(self.random_dataset, batch_size=self.num_samples,
                                                             shuffle=True, num_workers=4)

    def update_concept_examples(self, concept_examples):
        """
        Update Concept Examples and recompute the cav vector
        """
        # Update the concept examples
        self.concept_examples = concept_examples
        self.num_samples = concept_examples.shape[0]

        # Update the number of samples in the random dataloader
        self.random_dataloader = torch.utils.data.DataLoader(self.random_dataset, batch_size=self.num_samples,
                                                             shuffle=True, num_workers=4)
        # Recompute the cav
        self.recompute_cav()

    def recompute_cav(self):
        """
        Recompute the cav vector
        """
        self.cav = self.train_and_get_CAV()

    def get_sensitivity_on_input_example(self, input_class_example):
        """
        Compute the sensitivity of a given input example using the trained cav and computed partial-gradient
        """
        if type(self.cav)!=np.ndarray:
            # Compute the cav vector if it is undefined
            self.cav = self.train_and_get_CAV()

        # Compute the partial gradient vector for the given sample
        self.partial_gradient = self.get_partial_derivative_on_input_example(input_class_example)

        # Compute the sensitivity for the given sample
        sensitivity = np.dot(self.partial_gradient, self.cav.T)

        return sensitivity

    def get_partial_derivative_on_input_example(self, input_class_example):
        """
        Compute the partial gradient of the remainder-network's class output w.r.t the given input example in the intermediate activation-space
        """
        # Encode the input to the chosen activation space without flattening
        input_in_activation_space, _ = self.get_activation_space_points(input_class_example, flatten_to_numpy=False)

        # Zero and retain grads
        input_in_activation_space.requires_grad_()
        input_in_activation_space.retain_grad()
        self.remainder_network[0].zero_grad()
        self.remainder_network[1].zero_grad()

        # Extract output of the remainder-network from the given input example in the intermediate action space
        intermediate = self.remainder_network[0](input_in_activation_space) # Upto last FC layer
        intermediate = intermediate.flatten(start_dim=1)
        out = self.remainder_network[1](intermediate)[0,self.class_index] #Final FC layer

        # Extract the required gradient and flatten it
        out.backward()
        partial_gradient = input_in_activation_space.grad.flatten(start_dim=1)

        return partial_gradient.cpu().detach().numpy()

    def train_and_get_CAV(self, plot=False):
        """
        Train a linear classifier to distingusih between random samples and the given concept sample in the chosen activation-space.
        Returns the CAV unit vector orthogonal to the classier's decision boundary.
        """
        # Create random/concept samples dataset from state variables
        activations, activation_labels = self.create_CAV_dataset()

        # Train the classifier
        clf = LogisticRegression().fit(activations, activation_labels.flatten())

        # Extract the CAV unit vector orthogonal to the clf's decision boundary.
        cav = -1*clf.coef_/np.linalg.norm(clf.coef_)

        if plot:
            # Fit PCA to 2 dimensions
            pca = PCA(n_components=2)
            reduced_activations = pca.fit_transform(activations)

            # Get the reduced cav as a unit vector in the new 2D space
            reduced_cav_offset = pca.transform(cav)[0]
            reprojected_mean = pca.transform(np.zeros(cav.shape))[0]
            reduced_cav_scaled =  reduced_cav_offset - reprojected_mean
            reduced_cav = reduced_cav_scaled/np.linalg.norm(reduced_cav_scaled)

            # Plot the dataset and CAV
            scatter_plot_classes(reduced_activations, activation_labels, class_labels=[0,1], class_colors=['m','k'], legend=['concepts', 'negative'], vector=reduced_cav, vector_color='r')

        return cav

    def create_CAV_dataset(self):
        """
        Create and return a dataset of the concept examples and random examples in the chosen activation space.
        """
        # Sample random examples if negative_examples are not given
        if not self.use_known_negative_examples:
            self.negative_examples = self.get_random_examples()

        # Combine and shuffle the concept examples and random samples as one dataset
        dataset = [(c, 0) for c in self.concept_examples] +[(n, 1) for n in self.negative_examples]
        dataset = [(x.unsqueeze(0),y) for x,y in dataset] # Add batch dimension
        random.shuffle(dataset)

        return self.get_activation_space_points(dataset)

    def get_random_examples(self):
        """
        Sample random samples of from the random dataloader
        """
        return next(iter(self.random_dataloader))[0]

    def get_activation_space_points(self, data, flatten_to_numpy=True, dim_reduce_to=None):
        """
        Encode the given data to the chosen activation space, flatten and reduce dimensionality as desired.
        flatten_to_numpy : flattens the output tensors and converts them to numpy vectors if True
                           returns the unflattened torch-tensors if False.
        """
        # Encode the input data to the chosen activation spaces
        activations = []
        activation_labels = []
        for inputs, labels in data:
            inputs = inputs.to(device)
            batch_labels = labels
            batch_extraction = self.activation_extractor(inputs)
            activations.append(batch_extraction[self.layer_out])

            activation_labels.append(batch_labels)

        # Concatenate the output from batches
        activation_labels = np.vstack(activation_labels)

        if flatten_to_numpy:
            # Flatten samples in the activation space and convert to a numpy array
            activations = np.vstack([a.flatten(start_dim=1).cpu().detach().numpy() for a in activations])
        else:
            # Preserve samples in the activation-space as unflattened torch-tensors
            activations = torch.vstack(activations)

        if dim_reduce_to != None:
            # Reduce dimension if desired
            pca = PCA(n_components=dim_reduce_to)
            activations = pca.fit_transform(activations)
            # Return with the pca model to transform other vectors as desired.
            return (activations, activation_labels, pca)

        return (activations, activation_labels)

    def get_activation_extractor(self):
      """
      Extract and return the intermediate output of the model at the chosen activation-space
      """
      return create_feature_extractor(self.model, return_nodes={self.layer_out:self.layer_out})

    def get_remainder_network(self):
        """
        Makes a copy of the network layers after the starting layer and returns that 'remainder_network'
        (The intermediate model output of the starting layer is expected as input by the extracted remainder network)
        """
        # Stores the remainder network's child layers
        remainder_layers = []
        reached_starting_layer = False

        # Iterate through the model's children to extract the remainder network after the chosen sub-layer
        for name, module in self.model.named_children():
            if not reached_starting_layer:
                if name.startswith(self.layer_out):
                    # The remainder network starts after this layer
                    reached_starting_layer=True

                # Skip layers before the starting layer
                continue

            # Add layer to the remainder network
            remainder_layers.append(module)

        # Create and return a model from the extracted remainder network child layers (pretrained weights preserved)
        return (torch.nn.Sequential(*remainder_layers[:-1]), remainder_layers[-1])