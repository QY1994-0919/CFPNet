# Centralized Feature Pyramid for Object Detection

This repository contains the official PyTorch implementation of the following paper:

#### Centralized Feature Pyramid for Object Detection  

Yu Quan,  Dong Zhang, Liyan Zhang and Jinhui Tang  
Computer Science and Engineering, Nanjing University of Science and Technology   
https://arxiv.org/abs/2210.02093

## Abstract 

Visual feature pyramid has shown its superiority in both effectiveness and efficiency in a wide range of applications. However, the existing methods exorbitantly concentrate on the inter-layer feature interactions but ignore the intra-layer feature regulations, which are empirically proved beneficial. Although some methods try to learn a compact intra-layer feature representation with the help of the attention mechanism or the vision transformer, they ignore the neglected corner regions that are important for dense prediction tasks. To address this problem, in this paper, we propose a Centralized Feature Pyramid (CFP) for object detection, which is based on a globally explicit centralized feature regulation. Specifically, we first propose a spatial explicit visual center scheme, where a lightweight MLP is used to capture the globally long-range dependencies and a parallel learnable visual center mechanism is used to capture the local corner regions of the input images. Based on this, we then propose a globally centralized regulation for the commonly-used feature pyramid in a top-down fashion, where the explicit visual center information obtained from the deepest intra-layer feature is used to regulate frontal shallow features. Compared to the existing feature pyramids, CFP not only has the ability to capture the global long-range dependencies, but also efficiently obtain an all-round yet discriminative feature representation. Experimental results on the challenging MS-COCO validate that our proposed CFP can achieve the consistent performance gains on the state-of-the-art YOLOv5 and YOLOX object detection baselines.  

## The overall architecture

![The overall architecture](https://github.com/QY1994-0919/CFP-master/blob/main/assets/overall.png)<br>

## Qualitative results

![Qualitative results](https://github.com/QY1994-0919/CFPNet/blob/main/assets/results.png)<br>

## Model Weights<br>
 Here, we present weights of CFP with YOLOX as the baseline.<br>
 
| Model | size | mAP(%) | weights |
| :--- | :---: | :---: | ---: |
| CFP-s | 640 | 41.1 | [weight](https://pan.baidu.com/disk/main#/index?category=all&path=%2FCFP-main%2Fweights) | 
| CFP-m | 640 | 46.4 | [weight](https://pan.baidu.com/disk/main#/index?category=all&path=%2FCFP-main%2Fweights) |
| CFP-l | 640 | 49.4 | [weight](https://pan.baidu.com/disk/main#/index?category=all&path=%2FCFP-main%2Fweights) | 

## Installation<br>
  ### Install CFP-main from source<br>
  
  	git clone git@github.com:QY1994-0919/CFP-main.git         
    cd CFP-main    
    pip3 install -v -e .  # or  python3 setup.py develop   
   
  ### Prepare COCO dataset<br>

    cd CFP-main   
    ln -s /path/to/your/COCO ./datasets/COCO   
    
## Train:Reproduce our results on COCO by specifying -f:<br>

     python -m cfp.tools.train -f cfp-s -d 2 -b 16 --fp16 -o [--cache]
     python -m cfp.tools.train -f cfp-m -d 2 -b 16 --fp16 -o [--cache]
     python -m cfp.tools.train -f cfp-l -d 2 -b 16 --fp16 -o [--cache]
                                                                   
## Evaluation: support batch testing for fast evaluation:<br>
                                  
      python -m cfp.tools.eval -n  cfp-s -c cfp_s.pth -b 16 -d 2 --conf 0.001 [--fp16] [--fuse]
      python -m cfp.tools.eval -n  cfp-m -c cfp_s.pth -b 16 -d 2 --conf 0.001 [--fp16] [--fuse]
      python -m cfp.tools.eval -n  cfp-l -c cfp_s.pth -b 16 -d 2 --conf 0.001 [--fp16] [--fuse]
                            

# Acknowledgement<br>
 Thanks [YOLOv5](https://github.com/ultralytics/yolov5) and [YOLOX](https://arxiv.org/abs/2107.08430) teams for the wonderful open source project!

# Bibtex
If you find this work is useful for your research, please cite our paper using the following BibTeX[[pdf](https://github.com/QY1994-0919/CFPNet.git)][[arxiv](https://arxiv.org/abs/2210.02093)]:<br>


      @article{quan2022centralized,
      title={Centralized Feature Pyramid for Object Detection},
      author={Quan, Yu and Zhang, Dong and Zhang, Liyan and Tang, Jinhu},
      journal={arXiv preprint arXiv:2210.02093},
      year={2022}}

