## Training and Evaluation

### Training

We train our models with 4 H200 GPUs. The models can also be trained with 4xA100s/H100s. The 7B model requires 80G gpu cards, and 3B model requires 48G gpu cards. 

We release our trained models on [Huggingface](https://huggingface.co/datasets/fangruiz/struct2d). 

```bash
sh scripts/train.sh
```

We will add more settings under `scripts/`.


### Evaluation

We released the code, and we will release scripts of different benchmarks soon.


