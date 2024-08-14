# Spike-driven Transformer V2 for TPL on ImageNet

### Requirements

```python3
pytorch >= 2.0.0
cupy
spikingjelly == 0.0.0.0.12
```

### Results on Imagenet-1K

Pre-trained ckpts and training logs of 55M: [here](https://drive.google.com/drive/folders/12JcIRG8BF6JcgPsXIetSS14udtHXeSSx?usp=sharing).

### Data Prepare

ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```shell
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Finetune

> Please download caformer_b36_in21_ft1k.pth first following [PoolFormer](https://github.com/sail-sg/poolformer).

execute: \
  `bash main.sh`
