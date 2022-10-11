import matplotlib.pyplot as plt

import torch
from torch import ne, nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image

import numpy as np

architectures = {"vgg16": 25088,
                 "densenet121": 1024,
                 "alexnet": 9216}


def load_data(data_directory="./flowers"):

    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, validloader, testloader, train_data


def network_setup(architecture, device='gpu', learning_rate=0.001, hidden_layer_1=512, hidden_layer_2=256, dropout=0.5):

    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("Choose appropriate model (vgg16, densenet121, alexnet")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('inputs', nn.Linear(1024, hidden_layer_1)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('hl1', nn.Linear(hidden_layer_1, hidden_layer_2)),
        ('relu2', nn.ReLU()),
        ('hl2', nn.Linear(hidden_layer_2, 102)),
        ('outputs', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()

    return model, criterion, optimizer


def train_network(model, criterion, optimizer, epochs=5, print_every=5, device='gpu'):

    trainloader, validloader, testloader, train_data = load_data()

    steps = 0
    running_loss = 0

    print('Training starts')

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            if torch.cuda.is_available() and device == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                for inputs, labels in validloader:
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs, labels = inputs.to(
                            'cuda:0'), labels.to('cuda:0')
                        model.to('cuda:0')

                    with torch.no_grad():
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(
                            equals.type(torch.FloatTensor)).item()

                valid_lost = batch_loss / len(validloader)
                accuracy = accuracy / len(validloader)

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.5f}.. "
                      f"Validation loss: {valid_lost:.5f}.. "
                      f"Validation accuracy: {accuracy:.5f}")
                running_loss = 0

    print('Training ended')


def test_accuracy():

    trainloader, validloader, testloader, train_data = load_data()

    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test images is: %d%%' % (100 * correct / total))


def save_checkpoint(architecture, device):

    trainloader, validloader, testloader, train_data = load_data()
    model, criterion, optimizer = network_setup(architecture, device)

    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'structure': architecture,
                  'input_size': 1024,
                  'hl1': 512,
                  'hd2': 256,
                  'dropout': 0.2,
                  'output_size': 102,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer_dict': optimizer.state_dict()
                  }

    torch.save(checkpoint, 'checkpoint.pth')


def load_checkpoint(filepath, architecture, device):

    checkpoint = torch.load(filepath)
    structure = checkpoint['structure']
    inputs = checkpoint['input_size']
    hidden_layer_1 = checkpoint['hl1']
    hidden_layer_2 = checkpoint['hd2']
    dropout = checkpoint['dropout']
    output = checkpoint['output_size']

    model, _, _ = network_setup(architecture, device)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict = checkpoint['state_dict']
    model.load_optimizer_dict = checkpoint['optimizer_dict']

    return model


def process_image(image):

    img_pil = Image.open(image)

    processing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    img_tensor = processing(img_pil)

    return img_tensor


def predict(image_path, model, topk=5, device='gpu'):

    if torch.cuda.is_available() and device == 'gpu':
        model.to('cuda:0')

    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.float()

    with torch.no_grad():
        log_ps = model(img_tensor.cuda())

    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(topk)
    top_p, top_class = np.array(
        top_p.to('cpu')[0]), np.array(top_class.to('cpu')[0])

    idx_to_class = {x: y for y, x in model.class_to_idx.items()}

    top_classes = []
    for index in top_class:
        top_classes.append(idx_to_class[index])

    return list(top_p), list(top_classes)
