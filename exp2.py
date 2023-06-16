import os,shutil
import torch,argparse
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from nltk import edit_distance
import matplotlib.pyplot as plt
import torchvision.transforms as T
from strhub.models.base import BaseSystem
from strhub.data.utils import CharsetAdapter
from strhub.models.utils import create_model
from strhub.models.utils import parse_model_args
from strhub.data.module import SceneTextDataModule

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_dir', required=True ,help="Model checkpoint (like outputs/parseq/2022-09-17_16-52-32)")
    parser.add_argument('--model_weights', type=str,default=None)
    parser.add_argument('--data_root', default='/DATA/parseq/val')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vis_dir', type=str,default="bad_samples")
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    os.makedirs("test_outputs",exist_ok=True)
    
    limits = [90,80,70,50,25,0]

    date_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(os.path.join("test_outputs",date_time+"_pred.txt"),"w",encoding="utf-8") as file:
        file.write("path\tED\tpred\n")
    with open(os.path.join("test_outputs",date_time+"_gt.txt"),"w",encoding="utf-8") as file:
        file.write("path\tED\tlabel\n")
    if args.visualize:
        for limit in limits:
            if os.path.exists(args.vis_dir+"_"+str(limit)):
                shutil.rmtree(args.vis_dir+"_"+str(limit))
            os.mkdir(args.vis_dir+"_"+str(limit))
            with open(os.path.join(args.vis_dir+"_"+str(limit),"log_pred.txt"),"w",encoding="utf-8") as file:
                file.write("path\tED\tpred\n")
            with open(os.path.join(args.vis_dir+"_"+str(limit),"log_gt.txt"),"w",encoding="utf-8") as file:
                file.write("path\tED\tlabel\n")
    
    #with open("UrduGlyphs.txt","r",encoding="utf-8") as file:
    #    lines = file.readlines()
    #chars=""
    #for line in lines:
    #    chars+=line.strip("\n")
    #chars+= " "
    #charset_test = chars

    #kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    print("Loading trained model from ", str(args.trained_dir)," ...")
    model: BaseSystem = create_model(args.trained_dir, True)
    if args.model_weights:
        print("Loading weights from ", str(args.model_weights)," ...")
        checkpoint = torch.load(args.model_weights)
        model.load_state_dict(checkpoint['state_dict'])
    model = model.eval().to(args.device)
    #model.charset_adapter = CharsetAdapter(charset_test)

    #model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams

    print("Model HPs: ",hp)
    
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, "UrduGlyphs.txt", args.batch_size, args.num_workers, False,False,False, rotation=0)

    test_dataset = datamodule.test_dataloaders().dataset

    print("Number of samples: ",len(test_dataset))

    accuracy_arr = []

    total = 0
    ned = 0
    weighted_ED = 0
    label_length = 0
    for i,(img, gt) in enumerate(tqdm(test_dataset)): #, desc=f'{name:>{max_width}}'):
        img_squeezed = img.unsqueeze(0)
        #print(img.shape)
        img_squeezed = img_squeezed.to(args.device)
        logit = model.forward(img_squeezed)
        prob = logit.softmax(-1)
        pred, prob = model.tokenizer.decode(prob)
        pred = pred[0]
        pred = model.charset_adapter(pred)
        gt_no_spaces = str(gt).replace(" ","")
        pred_no_spaces = str(pred).replace(" ","")
        total += 1
        if len(gt) == 0 or len(pred) == 0:
            ED = 0
        else:
            ED = edit_distance(pred_no_spaces, gt_no_spaces) / max(len(pred_no_spaces), len(gt_no_spaces))
        acc = (1 - float(ED))*100
        accuracy_arr.append(acc)
        ned += ED
        weighted_ED += ED*len(gt)
        label_length += len(gt)
        with open(os.path.join("test_outputs",date_time+"_pred.txt"),"a",encoding="utf-8") as file:
            file.write("Number-" + str(i)+"\t|\t"+str(acc)+"\t|\t"+pred+"\t|\n")
        with open(os.path.join("test_outputs",date_time+"_gt.txt"),"a",encoding="utf-8") as file:
            file.write("Number-" + str(i)+"\t|\t"+str(acc)+"\t|\t"+gt+"\t|\n")
        
        if args.visualize:
            for limit in limits:
                if acc>=limit:
                    new_file = str(i)+".jpg"
                    img = img*0.5 + 0.5 # Undo normalisation
                    transform = T.ToPILImage()
                    img = transform(img)
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    img.save(os.path.join(args.vis_dir+"_"+str(limit),new_file))
                    with open(os.path.join(args.vis_dir+"_"+str(limit),"log_pred.txt"),"a",encoding="utf-8") as f:
                        f.write(new_file+"\t|\t"+str(acc)+"\t|\t"+pred+"\t|\n")
                    with open(os.path.join(args.vis_dir+"_"+str(limit),"log_gt.txt"),"a",encoding="utf-8") as f:
                        f.write(new_file+"\t|\t"+str(acc)+"\t|\t"+gt+"\t|\n")
                    break
                else:
                    continue
                
    
    mean_ned = 100 * (1 - ned / total)
    weighted_mean_ned = 100 * (1 - weighted_ED / float(label_length))

    print("Accuracy: ", mean_ned)
    print("Weighted Accuracy: ", weighted_mean_ned)
    print("Outputs written at ", os.path.join("test_outputs",date_time+".txt"))
    with open(os.path.join("test_outputs",date_time+".txt"),"a",encoding="utf-8") as file:
        file.write("Accuracy: " + str(mean_ned)+"\n")
        file.write("Weighted Accuracy: " + str(weighted_mean_ned))
    plt.hist(accuracy_arr)
    plt.savefig(os.path.join("test_outputs",date_time+".png"))
    print("Histogram saved at ",os.path.join("test_outputs",date_time+".png"))

if __name__ == '__main__':
    main()
