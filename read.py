# Inference script for the model

import argparse
import torch,os
from PIL import Image
from tqdm import tqdm
from strhub.models.base import BaseSystem
from strhub.models.utils import create_model
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import parse_model_args


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_dir', required=True ,help="Model checkpoint (like outputs/parseq/2022-09-17_16-52-32)")
    parser.add_argument('--model_weights', type=str,default=None)
    parser.add_argument('--image_path',  type=str,required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_flip', action='store_true',)
    parser.add_argument('--result_path',  type=str, default='read_output.txt')
    args, unknown = parser.parse_known_args() 
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')
    
    if os.path.exists(args.result_path):
        os.remove(args.result_path)

    print("Loading trained model from ", str(args.trained_dir)," ...")
    model: BaseSystem = create_model(args.trained_dir, True)
    if args.model_weights:
        print("Loading weights from ", str(args.model_weights)," ...")
        checkpoint = torch.load(args.model_weights)
        model.load_state_dict(checkpoint['state_dict'])
    model = model.eval().to(args.device)
    
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    #for args.image_path in args.images:
    # Load image and prepare for input
    
    images_to_read = []
    if not os.path.exists(args.image_path):
        exit()
    if os.path.isfile(args.image_path):
        images_to_read = [args.image_path]
    elif os.path.isdir(args.image_path):
        for images in os.listdir(args.image_path):
            images_to_read.append(os.path.join(args.image_path,images))
    
    for image_path in tqdm(images_to_read):
        image = Image.open(image_path).convert('RGB')
        if not args.no_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = img_transform(image).unsqueeze(0).to(args.device)
        p = model(image).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        image_path = image_path.split("/")[-1]
        with open(args.result_path, 'a') as f:
            f.write(image_path+" "+pred[0]+"\n")
    print(f'Transcription stored in ', args.result_path)
if __name__ == '__main__':
    main()
