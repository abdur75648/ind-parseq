import os,shutil
import numpy as np
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

def allign_two_strings(x:str, y:str, pxy:int=1, pgap:int=1):
    """
    Source: https://www.geeksforgeeks.org/sequence-alignment-problem/
    """
    
    i = 0
    j = 0
    m = len(x)
    n = len(y)
    dp = np.zeros([m+1,n+1], dtype=int)
    dp[0:(m+1),0] = [ i * pgap for i in range(m+1)]
    dp[0,0:(n+1)] = [ i * pgap for i in range(n+1)]
 
    # calculating the minimum penalty
    i = 1
    while i <= m:
        j = 1
        while j <= n:
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1] + pxy,
                                dp[i - 1][j] + pgap,
                                dp[i][j - 1] + pgap)
            j += 1
        i += 1
     
    # Reconstructing the solution
    l = n + m   # maximum possible length of the alignment
    i = m
    j = n
     
    xpos = l
    ypos = l
 
    # Final answers for the respective strings
    xans = np.zeros(l+1, dtype=int)
    yans = np.zeros(l+1, dtype=int)
 
    while not (i == 0 or j == 0):
        #print(f"i: {i}, j: {j}")
        if x[i - 1] == y[j - 1]:       
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
        elif (dp[i - 1][j - 1] + pxy) == dp[i][j]:
         
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
         
        elif (dp[i - 1][j] + pgap) == dp[i][j]:
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord('_')
            xpos -= 1
            ypos -= 1
            i -= 1
         
        elif (dp[i][j - 1] + pgap) == dp[i][j]:       
            xans[xpos] = ord('_')
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            j -= 1
         
 
    while xpos > 0:
        if i > 0:
            i -= 1
            xans[xpos] = ord(x[i])
            xpos -= 1
        else:
            xans[xpos] = ord('_')
            xpos -= 1
     
    while ypos > 0:
        if j > 0:
            j -= 1
            yans[ypos] = ord(y[j])
            ypos -= 1
        else:
            yans[ypos] = ord('_')
            ypos -= 1
 
    # Since we have assumed the answer to be n+m long,
    # we need to remove the extra gaps in the starting
    # id represents the index from which the arrays
    # xans, yans are useful
    id = 1
    i = l
    while i >= 1:
        if (chr(yans[i]) == '_') and chr(xans[i]) == '_':
            id = i + 1
            break
         
        i -= 1
 
    # # Printing the final answer
    # print(f"Minimum Penalty in aligning the genes = {dp[m][n]}")
    # print("The aligned genes are:")   
    # X
    i = id
    x_seq = ""
    while i <= l:
        x_seq += chr(xans[i])
        i += 1
    # print(f"X seq: {x_seq}")
 
    # Y
    i = id
    y_seq = ""
    while i <= l:
        y_seq += chr(yans[i])
        i += 1
    # print(f"Y seq: {y_seq}")
    
    return x_seq, y_seq

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_dir', required=True ,help="Model checkpoint (like outputs/parseq/2022-09-17_16-52-32)")
    parser.add_argument('--model_weights', type=str,default=None)
    parser.add_argument('--data_root', default='/DATA/parseq/val')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--threshold', type=int, default=80)
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
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
    
    with open("UrduGlyphs.txt","r",encoding="utf-8") as file:
       lines = file.readlines()
    chars=[]
    for line in lines:
       chars.append(line.strip("\n"))
    # chars.append(" ")

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
    
    # Calculate character-wise accuracy between the two strings
    total_occurence = {}
    correct_occurence = {}

    # Add all alphabets to total_occurence with value 0
    for char in chars:
        total_occurence[char] = 0
        correct_occurence[char] = 0
    
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
        if len(gt_no_spaces) == 0 or len(pred_no_spaces) == 0:
            ED = 0
        else:
            ED = edit_distance(pred_no_spaces, gt_no_spaces) / max(len(pred_no_spaces), len(gt_no_spaces))
        
        gt_aligned,pred_aligned = allign_two_strings(gt_no_spaces, pred_no_spaces)
        
        # Count total occurence of each alphabet in both strings
        for i in range(len(gt_aligned)):
            total_occurence[gt_aligned[i]] += 1
            # Now check if the character is correct in the prediction
            if gt_aligned[i] == pred_aligned[i]:
                correct_occurence[gt_aligned[i]] += 1
        
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
    
    Accuracy = {}
    for char in chars:
        if total_occurence[char] != 0:
            Accuracy[char] = 100*correct_occurence[char]/total_occurence[char]
    
    sorted_accuracy = sorted(Accuracy.items(), key=lambda x: x[1], reverse=True)
    
    import pandas as pd
    df = pd.DataFrame(columns=["Alphabet", "Accuracy"])
    for key, value in sorted_accuracy:
        if value != 0:
            print(f"Accuracy of {key}: {value:.2f}")
            df = df.append({"Alphabet": key, "Accuracy": value}, ignore_index=True)
    
    df.to_csv("Character-wise-accuracy.csv", index=False)
    
    plt.figure(figsize=(10, 5))
    plt.bar(df["Alphabet"], df["Accuracy"])
    plt.xlabel("Alphabet")
    plt.ylabel("Accuracy")
    plt.savefig("Character-wise-accuracy.png")
    

if __name__ == '__main__':
    main()