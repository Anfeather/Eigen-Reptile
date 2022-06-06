# Robust Meta-learning with Noise via Introspective Eigen-Reptile

## Dataset Link
https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk/view

We evaluate our method on a popular few-shot classification dataset: Mini-ImageNet. The Mini-Imagenet dataset contains 100 classes, each with 600 images. We divide the dataset into three disjoint subsets: meta-training set, meta-validation set, and meta-testing set with 64 classes, 16 classes and 20 classes, respectively.

## Requirements
numpy==1.18.1\
tensorflow-gpu==1.13.1\
python==3.7.6


## Training & Evaluation

5-way 1-shot 

nohup python3 -u run_miniimagenet.py --ratio 0.0 --shots 1 --inner-batch 10 --inner-iters 7 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m15t_train --transductive > mtest15_train.log 2>&1 &

5-way 5-shot

nohup python3 -u run_miniimagenet.py --ratio 0.0 --inner-batch 10 --inner-iters 7 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 15 --eval-iters 50 --learning-rate 0.0005 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m55_train_test --transductive > mtest55_train_test.log 2>&1 &

For symmetric noise, you should uncomment line 130 at first. For asymmetric noise, you should run create_noise.py at first to generate the corresponding dataset. Note that we change the hyperparameters to make all methods get similar results when p=0 to examine the robustness of all methods.

nohup python3 -u run_miniimagenet.py --ratio 0.5 --shots 1 --inner-batch 10 --inner-iters 7 --meta-step 1 --meta-batch 5 --meta-iters 10000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m15t_train --transductive > mtest15_train_anoise.log 2>&1 &
## Results

Our model achieves the following performance on :

### [Few-shot Classification on Mini-Imagenet]

| Model name         | 5-way 1-shot  | 5-way 5-shot |
| ------------------ |---------------- | -------------- |
| Reptile |49.97 ± 0.32\% | 65.99 ± 0.58\% |
| Eigen-Reptile (32)     |51.80 ± 0.90\%  | 68.10 ± 0.50%  |
| Eigen-Reptile (64)     |53.25 ± 0.45\%  | 69.55 ± 0.65%  |


### [Noisy Few-shot Classification on Corrupted Mini-Imagenet (Symmetric Noise)]

| Model name         | p=0.0  | p=0.1 | p=0.2  | p=0.5 |
| ------------------ |---------------- | -------------- |---------------- | -------------- |
| Reptile |47.64% | 46.08% | 43.49 %| 23.33%|
| Eigen-Reptile      |47.87\%  | 47.18%  | 45.01\%  | 27.23%  |
| Eigen-Reptile+ISPL      |47.26\%  | 47.20%  | 45.49\%  | 28.68%  |

### [Noisy Few-shot Classification on Corrupted Mini-Imagenet (Asymmetric Noise)]

| Model name         |  p=0.1 | p=0.2  | p=0.5 |
| ------------------ |---------------- | -------------- |---------------- |
| Reptile | 47.30% | 45.51 %| 42.03%|
| Eigen-Reptile      | 47.42%  | 46.50\%  | 42.29%  |
| Eigen-Reptile+ISPL      | 47.24%  | 46.83\%  | 43.71%  |








