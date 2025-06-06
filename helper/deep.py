import torch

def detect_device():
    """Automatically detects if you have a cuda enabled GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def encode_batchwise(dataloader, model, device):
    """Utility function for embedding the whole data set in a mini-batch fashion"""
    embeddings = []
    labels = []
    for batch, batch_labels, ids in dataloader:
        batch_data = batch.to(device)
        embeddings.append(model.encode(batch_data).detach().cpu())
        labels.append(batch_labels)
    return torch.cat(embeddings, dim=0).numpy(), torch.cat(labels, dim=0)

def decode_batchwise(dataloader, model, device):
    """Utility function for decoding the whole data set in a mini-batch fashion"""
    decodings = []
    for batch, _, _ in dataloader:
        batch_data = batch.to(device)
        decodings.append(model(batch_data).detach().cpu())
    return torch.cat(decodings, dim=0).numpy()

def get_train_and_testloader(data, labels, batch_size):
    # create a Dataloader to train the autoencoder in mini-batch fashion
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(data, labels, torch.arange(0, len(labels)))),
        batch_size=batch_size,
        # sample random mini-batches from the data
        shuffle=True,
        drop_last=False,
    )
    # create a Dataloader to test the autoencoder in mini-batch fashion (Important for validation)
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(data, labels, torch.arange(0, len(labels)))),
        batch_size=batch_size,
        # Note that we deactivate the shuffling
        shuffle=False,
        drop_last=False,
    )
    return trainloader, testloader

def pretrain_model(
    device,
    pretrained_model_path,
    model,
    data,
    trainloader,
    optimizer,
    loss_fn,
    training_iterations=30,
    verbose=True,
):
    # load model to device
    model.to(device)

    # start training
    i = 0
    # training loop
    while i < training_iterations:
        for batch, _, _ in trainloader:
            # load batch on device
            batch_data = batch.to(device)

            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch_data)

            # reset gradients from last iteration
            optimizer.zero_grad()
            # calculate gradients and reset the computation graph
            loss.backward()
            # update the internal params (weights, etc.)
            optimizer.step()

        if verbose:
            print(f"Iteration {i+1}/{training_iterations} - Reconstruction loss: {loss.item():.6f}")
        i += 1
    # save model
    torch.save(model.state_dict(), pretrained_model_path)
    return model

def load_pretrained_model(model, pretrained_model_path, device, work_on_copy=False):
    state_dict = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(state_dict)
    if hasattr(model, "fitted"):
        model.fitted = True
    model.work_on_copy = work_on_copy
    return model