import torch
import torch.nn.functional as F

def pytorch_train(model, dataloader, device, optimizer, criterion, epoch=0):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

def pytorch_test(model, device, dataloader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images.to(device))

            _, predicted = torch.max(outputs.data, 1)
            test_loss += F.nll_loss(outputs, labels.to(device))  # sum up batch loss
            correct += predicted.eq(labels.to(device)).sum().item()

    test_loss /= len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset), accuracy))

    return test_loss, accuracy


def pytorch_predict(model, device, dataloader, return_labels=True):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images.to(device))
            for out in outputs:
                preds.append(out)
            if return_labels == True:
                for targ in labels.to(device):
                    targets.append(targ)

    if return_labels == True:
        return preds, targets
    else:
        return preds