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
    #parser.add_argument('--data_root', default='/DATA/parseq/val')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--threshold', type=int, default=80)
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    
    """
    os.makedirs("test_outputs",exist_ok=True)

    date_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(os.path.join("test_outputs",date_time+"_pred.txt"),"w",encoding="utf-8") as file:
        file.write("path\tED\tpred\n")
    with open(os.path.join("test_outputs",date_time+"_gt.txt"),"w",encoding="utf-8") as file:
        file.write("path\tED\tlabel\n")
    if args.visualize:
        if os.path.exists("bad_samples"):
            shutil.rmtree("bad_samples")
        os.mkdir("bad_samples")
        bad_file_pred = "bad_samples/log_pred.txt"
        with open(bad_file_pred,"w",encoding="utf-8") as file:
            file.write("path\tED\tpred\n")
        bad_file_gt = "bad_samples/log_gt.txt"
        with open(bad_file_gt,"w",encoding="utf-8") as file:
            file.write("path\tED\tlabel\n")
    
    #with open("UrduGlyphs.txt","r",encoding="utf-8") as file:
    #    lines = file.readlines()
    #chars=""
    #for line in lines:
    #    chars+=line.strip("\n")
    #chars+= " "
    #charset_test = chars

    #kwargs.update({'charset_test': charset_test})
    """
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
    for data_root in ['/DATA/IIITH dataset/lmdb_old','/DATA/IIITH dataset/lmdb_new','/DATA/parseq/val']:
        print("="*50)
        print(data_root)
        datamodule = SceneTextDataModule(data_root, '_unused_', hp.img_size, hp.max_label_length, "UrduGlyphs.txt", args.batch_size, args.num_workers, False,False,False, rotation=0)
        test_dataset = datamodule.test_dataloaders().dataset
        print("Number of samples: ",len(test_dataset))
        #accuracy_arr = []
        total = 0
        ned = 0
        weighted_ED = 0
        label_length = 0
        ned_no_spaces=0
        weighted_ED_no_spaces = 0
        label_length_no_spaces = 0
        for i,(img, gt) in enumerate(tqdm(test_dataset)): #, desc=f'{name:>{max_width}}'):
            img_squeezed = img.unsqueeze(0)
            #print(img.shape)
            img_squeezed = img_squeezed.to(args.device)
            logit = model.forward(img_squeezed)
            prob = logit.softmax(-1)
            pred, prob = model.tokenizer.decode(prob)
            pred = pred[0]
            pred = model.charset_adapter(pred)
            total += 1
            gt_no_spaces = str(gt).replace(" ","")
            pred_no_spaces = str(pred).replace(" ","")
            if len(gt) == 0 or len(pred) == 0 or len(gt_no_spaces) == 0 or len(pred_no_spaces) == 0:
                ED = 0
                ED_no_spaces = 0
            else:
                ED = edit_distance(pred, gt) / max(len(pred), len(gt))
                ED_no_spaces = edit_distance(pred_no_spaces, gt_no_spaces) / max(len(pred_no_spaces), len(gt_no_spaces))
            #accuracy_arr.append(acc)
            ned += ED
            ned_no_spaces += ED_no_spaces
            weighted_ED += ED*len(gt)
            weighted_ED_no_spaces += ED_no_spaces*len(gt_no_spaces)
            label_length += len(gt)
            label_length_no_spaces += len(gt_no_spaces)
            #with open(os.path.join("test_outputs",date_time+"_pred.txt"),"a",encoding="utf-8") as file:
                #file.write("Number-" + str(i)+"\t|\t"+str(acc)+"\t|\t"+pred+"\t|\n")
            #with open(os.path.join("test_outputs",date_time+"_gt.txt"),"a",encoding="utf-8") as file:
                #file.write("Number-" + str(i)+"\t|\t"+str(acc)+"\t|\t"+gt+"\t|\n")
            
            """
            if args.visualize:
                if acc<args.threshold:
                    new_file = str(i)+".jpg"
                    img = img*0.5 + 0.5 # Undo normalisation
                    transform = T.ToPILImage()
                    img = transform(img)
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    img.save(os.path.join("bad_samples",new_file))
                    with open(bad_file_pred,"a",encoding="utf-8") as f:
                        f.write(new_file+"\t|\t"+str(acc)+"\t|\t"+pred+"\t|\n")
                    with open(bad_file_gt,"a",encoding="utf-8") as f:
                        f.write(new_file+"\t|\t"+str(acc)+"\t|\t"+gt+"\t|\n")
            """
        
        mean_ned = 100 * (1 - ned / total)
        weighted_mean_ned = 100 * (1 - weighted_ED / float(label_length))
        mean_ned_no_spaces = 100 * (1 - ned_no_spaces / total)
        weighted_mean_ned_no_spaces = 100 * (1 - weighted_ED_no_spaces / float(label_length_no_spaces))
        print("="*10,"With spaces","="*10)
        print("Accuracy: ", mean_ned)
        print("Weighted Accuracy: ", weighted_mean_ned)
        print("="*10,"Without spaces","="*10)
        print("Accuracy: ", mean_ned_no_spaces)
        print("Weighted Accuracy: ", weighted_mean_ned_no_spaces)
        print("="*50)
    #print("Outputs written at ", os.path.join("test_outputs",date_time+".txt"))
    #with open(os.path.join("test_outputs",date_time+".txt"),"a",encoding="utf-8") as file:
        #file.write("Accuracy: " + str(mean_ned)+"\n")
        #file.write("Weighted Accuracy: " + str(weighted_mean_ned))
    #plt.hist(accuracy_arr)
    #plt.savefig(os.path.join("test_outputs",date_time+".png"))
    #print("Histogram saved at ",os.path.join("test_outputs",date_time+".png"))

if __name__ == '__main__':
    main()
