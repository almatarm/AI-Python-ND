import torch
import json
from model_helper import Phase
import model_helper as mh
import args_parser

if __name__ == '__main__':
  # Parse Argument
  args = args_parser.parse_args(Phase.valid)
  print(args)

  # Use GPU if it's available
  device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
  print('### Using device: ',  device)

  # Label Mapping
  with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

  # Loading the model
  probs, classes = mh.predict_from_disk(device, args.image_path, args.checkpoint, args.top_k)
  flowers_names = [cat_to_name[cat] for cat in classes]
  print()
  print("PROBABILITY   FLOWER NAME")
  print("===========   ====================================")
  for i in range(args.top_k):
    print('{:5.2f}%        {}'.format(probs[0][i]*100, flowers_names[i].upper()))
  