#!/bin/bash 

#SBATCH -J train  # 作业名为 test

#SBATCH -p defq  # 提交到 defq 队列

#SBATCH -N 1     # 使用 1 个节点

#SBATCH --ntasks-per-node=8  # 每个节点开启 6 个进程

#SBATCH --cpus-per-task=2    # 每个进程占用一个 cpu 核心

#SBATCH -t 7-24:00:00 # 任务最大运行时间是 10 分钟 (非必需项) 48:00:00


#SBATCH --gres=gpu:2    # 如果是gpu任务需要在此行定义gpu数量,此处为1

module load cuda11.0/toolkit/11.0.3

#python tools/demo.py image -f exps/default/yolox_s.py -c ./weights/cfp_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
python tools/train.py -f exps/example/custom/cfp_s.py -d 1 -b 16 --fp16 -o -c ./weights/cfp_s.pth
#python tools/train.py -f exps/example/custom/cfp_s.py -d 2 -b 16 --fp16 -o -c ./YOLOX_outputs/l_CCF/latest_ckpt.pth --resume -e 10



