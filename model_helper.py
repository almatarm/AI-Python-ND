import enum
import torch
from torch import optim
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

class Phase(enum.Enum):
    train = 'train'
    valid = 'valid'
    test = 'test'
  
def load_data(phase = Phase.train, batch_size = 64):
    data_dir = 'flowers' + '/' + phase.value
    
    # The input data is resized to 224x224 pixels as required by the pre-trained networks.
    nInputs = 224

    # For the means, it's [0.485, 0.456, 0.406] and for the standard deviations 
    # [0.229, 0.224, 0.225], calculated from the ImageNet images. 
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
    
    if(phase == Phase.train):
        # Apply transformations such as random scaling, cropping, and flipping.
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(nInputs),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Resize then crop the images to the appropriate size.
        data_transforms = transforms.Compose([
            transforms.Resize(nInputs),
            transforms.CenterCrop(nInputs),
            transforms.ToTensor(),
            normalize
        ])
    
    # Load the datasets with ImageFolder
    dataset = datasets.ImageFolder(
        data_dir,
        transform=data_transforms
    )
                    
    # Using the image datasets and the trainforms, define the dataloaders
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size, 
                                         shuffle=True if phase == Phase.train else False) 
    return dataset, loader, data_transforms
    
    
def build_model(arch, nHiddens = 256, nOutputs=102, pDropout=0.2, lr=0.003):
    if arch == 'densenet':
        model = models.densenet121(pretrained=True)
        nFeatures = 1024
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        nFeatures = 9216
    else:
        model = models.vgg16(pretrained=True)
        nFeatures = 25088

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad= False

    model.classifier = nn.Sequential(
        nn.Linear(nFeatures, nHiddens),
        nn.ReLU(),
        nn.Dropout(pDropout),
        nn.Linear(nHiddens, nOutputs),
        nn.LogSoftmax(dim=1)
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    return model, optimizer, criterion
    
def train_model(device, model, trainloader, optimizer, criterion, validloader, epochs = 5, stop_accuracy = .95, 
                verbose = False, test_every = 1):
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % test_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/test_every:.3f}.. "
                          f"Valid loss: {test_loss/len(validloader):.3f}.. "
                          f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
                if (accuracy/len(validloader)) >= stop_accuracy:
                    print('Accurecy Reached: ', accuracy/len(validloader), '>=', stop_accuracy)
                    return epoch
    return epochs

def test_model(device, model, criterion, testloader):
    test_loss = 0
    accuracy = 0
    pAccuracy = 0;
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            pAccuracy = accuracy/len(testloader) * 100
            print(f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
    return pAccuracy

def save_model(model, arch, dataset, optimizer, epoch, filepath='checkpoint.pth', nHiddens = 256, nOutputs=102, pDropout=0.2, lr=0.003):
    model.class_to_idx = dataset.class_to_idx
    checkpoint = {'arch': arch,
                  'nHiddens': nHiddens,
                  'nOutputs': nOutputs,
                  'pDropout': pDropout,
                  'lr': lr,
                  'class_to_idx': dataset.class_to_idx,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': epoch
                 }
    torch.save(checkpoint, filepath)

def load_model(filepath='checkpoint.pth'):
    cp = torch.load(filepath)
    arch = cp['arch']
    nHiddens = cp['nHiddens']
    nOutputs = cp['nOutputs']
    pDropout = cp['pDropout']
    lr = cp['lr']
    epoch = cp['epoch']
    model, optimizer, criterion = build_model(arch, 
                                              nHiddens = nHiddens, 
                                              nOutputs=nOutputs, 
                                              pDropout=pDropout, 
                                              lr=lr)
    model.load_state_dict(cp['model_state_dict'])
    optimizer.load_state_dict(cp['optimizer_state_dict'])
    model.class_to_idx = cp['class_to_idx']

    return model, optimizer, criterion, epoch;

def process_image(imagepath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    image = Image.open(imagepath)
    valid_dataset, validloader, valid_transforms = load_data(Phase.valid)
    image = valid_transforms(image)
    return image

def predict(device, image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    
    # Prepare image
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)
    image.device
    
    # You need to convert from these indices to the actual class labels using class_to_idx 
    # Make sure to invert the dictionary so you get a mapping from index to class as well.
    idx_to_class = {k:v for v,k in model.class_to_idx.items()} #inverted
    
    
    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        probs, class_idxs = ps.topk(topk, dim=1)
        class_idxs = class_idxs.cpu().numpy()
        classes = [idx_to_class[idx] for idx in class_idxs[0]]
         
        return probs, classes

def predict_from_disk(device, image_path, model_path='checkpoint.pth', topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model = load_model(model_path)[0]
    return predict(device, image_path, model, topk)