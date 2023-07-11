import os,lmdb,six,random
import shutil
from PIL import Image

# if os.path.exists("vis_data"):
#     shutil.rmtree("vis_data")
# os.makedirs("vis_data")

data_dir = "/home/abdur/scratch/ihtr_data_lmdb/malayalam/train/real"
env = lmdb.open(data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

with env.begin(write=False) as txn:
    nSamples = int(txn.get('num-samples'.encode()))
    print("Number of samples: ",nSamples)

def get_item(index):
    assert index < nSamples, 'index range error'
    with env.begin(write=False) as txn:
        # Label
        label_key = 'label-%09d'.encode() % index
        #print("Label key: ",txn.get(label_key))
        label = txn.get(label_key).decode('utf-8')
        # Image
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        #img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return (img, label)

for i in range(50):
    idx = random.randint(1,nSamples)
    img,label = get_item(idx)
    # with open("vis_data/label.txt","a",encoding="utf-8") as file:
    #     file.write("vis_data/"+str(idx)+".png\t-\t"+label+"\n")
    # img.save("vis_data/"+str(idx)+".png")