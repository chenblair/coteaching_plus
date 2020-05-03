#!/bin/bash

#for r in 0.2 0.5 0.65 0.8
for r in 0.2 0.3 0.45
do
	#python3 main_mnist.py --name coteachingp --dataset mnist --noise_type pairflip --noise_rate $r --result_dir results --n_epoch 50 --no-gamblers
	python3 main.py --name nllpair --dataset cifar10 --noise_rate $r --result_dir results --n_epoch 100 --noise_type pairflip 
done
# python3 main.py --name lq --dataset cifar10 --noise_type symmetric --noise_rate 0.8 --result_dir results --n_epoch 100 --gamblers --q 0.7
# python3 main_mnist.py --name coteaching --dataset mnist --noise_type symmetric --noise_rate $r --result_dir results --n_epoch 50 --no-gamblers
# python3 main.py --name coteaching --dataset cifar10 --noise_type symmetric --noise_rate $r --result_dir results --n_epoch 100 --no-gamblers
