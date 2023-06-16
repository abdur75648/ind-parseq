

# Parseq Model For Indian Languages

## Training
```CUDA_AVAILABLE_DEVICES=2,3,4,5,6,7 python3 train.py pretrained=outputs/parseq/2022-09-17_16-52-32/ trainer.devices=[2,3,4,5,6,7] learning_rate=0.0001```

## HPC-Training
```CUDA_AVAILABLE_DEVICES=0 python3 train.py trainer.devices=[0]```

## Fine-tuning
```CUDA_AVAILABLE_DEVICES=5,6 python3 train.py trainer.devices=[5,6] dataset=real data.root_dir=/home/abdur/scratch/ihtr_data_lmdb/malayalam/ model.lr=1e-5 model.batch_size=128 trainedmodel=outputs/parseq/hindi_pretrained/vit_hi_epoch_82.ckpt remove_head=true model.charset_file=MalayalamGlyphs.txt trainer.val_check_interval=100```

## Test
```CUDA_AVAILABLE_DEVICES=0 python3 test.py  --trained_dir outputs/parseq/2022-09-19_11-38-16/ --model_weights outputs/parseq/2022-09-19_11-38-16/checkpoints/last.ckpt  --data_root data/train/real_distorted/ --threshold 80 --visualize```

### Configs To Take Care of
* If langauge/dataset changes:
    * data.root_dir
    * dataset (real/synth)
    * data.charset_file
    * flip_left_right (if language is urdu)
* If modality changes (line-level or world-level):
    * remove_whitespace (true/false)
    * model.img_size
    * max_label_length
* If num_GPU>1:
    * traine.devices

## Setup For HPC-IITD
* ```module load apps/pytorch/1.10.0/gpu/intelpython3.7```

Create a venv (if not already done)
* ```python3 -m venv parseq_env```

Activate the venv
* ```source parseq_env/bin/activate```

Install Libraries
* ```pip install -r requirements.txt```
