import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, data_dir, num_epochs=10, return_best_test_set_model=False):
        """
        Trains a given model for some criterion, optimizer, scheduler and dataloaders.
        return_best_test_set_model : Saves and return the model checkpoint with highest accuracy on the test set if set to True
                                     Otherwise, return the final model after num_epochs of training
        """
        # Best model accuracy and filepath
        best_acc = 0.0
        best_model_ckpt = os.path.join(data_dir, 'model.ckpt')

        for epoch in range(num_epochs):
            print('Epoch: ', epoch,'/',(num_epochs - 1))
            print('------------------------------------')

            for split in ['train', 'test']:
                # Set mode to train/eval
                if split=='train':
                    model.train()
                else:
                    model.eval()

                # Variables for loss and accuracy statistics computation
                running_loss = 0.0
                running_corrects = 0

                # Iterate over mini batches.
                for inputs, labels in dataloaders[split]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero gradients if training
                    if split=='train':
                        optimizer.zero_grad()

                    # Compute Loss on minibatch
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Compute gradients and take a step of gradient descent for the minibatch if training
                    if split=='train':
                      loss.backward()
                      optimizer.step()

                    # Compute statistics on minibatch
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)


                # Print Statistics
                epoch_loss = running_loss / dataset_sizes[split]
                epoch_acc = float(running_corrects) / dataset_sizes[split]
                print(split+' loss: ', epoch_loss, ' ', split+ ' accuracy: ', epoch_acc)


                # Save the model that achives the best test-set accuracy
                if return_best_test_set_model and split == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_ckpt)


        # load the best model weights and return if desired
        if return_best_test_set_model:
            model.load_state_dict(torch.load(best_model_ckpt))

        return model

def eval_model(model, test_dataloader, test_dataset_size):
        """
        Evaluates and returns the accuracy a given model on the test set.
        """
        # Set mode to eval
        model.eval()

        # Variables for loss and accuracy statistics computation
        running_corrects = 0

        # Iterate over mini batches.
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Make predictions on minibatch
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Compute statistics on minibatch
            running_corrects += torch.sum(preds == labels.data)

        # Compute Final Statistics
        test_acc = float(running_corrects) / test_dataset_size

        return test_acc