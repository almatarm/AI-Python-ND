import torch
from model_helper import Phase
import model_helper as mh
import args_parser
from workspace_utils import active_session

if __name__ == '__main__':
  # Parse Argument
  args = args_parser.parse_args(Phase.train)
  print(args)

  # Use GPU if it's available
  device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
  print('### Using device: ',  device)

  # Loading the data
  print('### Loading data')
  train_dataset, trainloader, train_transforms = mh.load_data(Phase.train)
  test_dataset, testloader, test_transforms = mh.load_data(Phase.test)

  # Building Model
  print('### Building the model')
  arch = args.arch
  nHiddens = args.hidden_units
  nOutputs = 102
  pDropout = 0.2
  lr = args.learning_rate
  model, optimizer, criterion = mh.build_model(arch, nHiddens = nHiddens, nOutputs=nOutputs, pDropout=pDropout, lr=lr)
  model.to(device)

  # Train Model
  print('### Trainging the model')
  stop_accuracy = args.accurecy / 100.0
  with active_session():
    epoch = mh.train_model(device, model, trainloader, optimizer, criterion, testloader, verbose = True, stop_accuracy=stop_accuracy)

  # Testing Network 
  print('### Testing the model')
  accuracy = mh.test_model(device, model, criterion, testloader)
  print(f"Accuracy:  {accuracy:.2f}%")

  # Saving the model
  print('### Saving Model')
  filepath = args.save_dir
  mh.save_model(model, arch, train_dataset, optimizer, epoch, filepath, nHiddens = nHiddens, nOutputs=nOutputs, pDropout=pDropout, lr=lr)
  
  print('Model Saved to: ', filepath)