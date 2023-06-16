gt_file = "/home/abdur/scratch/ihtr_raw_data/train_data/telugu/train.txt"
with open(gt_file,"r",encoding="utf-8") as f:
    lines = f.readlines()

lines = [line.strip().split()[1] for line in lines]

# A set to store unique characters
bad_chars = set()

# Iterate over each line and add unique characters to the set and write to "UrduGlyphs.txt"
for line in lines:
    for char in line:
        bad_chars.add(char)

with open("TeluguGlyphs.txt","w",encoding="utf-8") as f:
    for char in bad_chars:
        f.write(char + "\n")