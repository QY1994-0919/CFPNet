# Centralized Feature Pyramid for Object Detection

Centralized Feature Pyramid (CFP) is based on a globally explicit centralized feature regulation for object detection. CFP not only has the ability to capture the
global long-range dependencies, but also efficiently obtain an all-round yet discriminative feature representation.<br>  

Experimental results on the challenging MS-COCO validate that our proposed CFP can achieve the consistent performance gains on the state-of-the-art YOLOv5 and YOLOX object detection baselines.<br>  

# Introduction
-----
Object detection is one of the most fundamental yet challenging research tasks in the community of computer vision, which aims to predict a unique bounding box for each object of the input image that contains not only the location but also the category information. In the past few years, this task has been extensively developed and applied to a wide range of potential applications, e.g., autonomous driving and computer-aided diagnosis. The successful object detection methods are mainly based
on the Convolutional Neural Network (CNN) as the backbone followed with a two-stage (e.g., Fast/Faster R-CNN)or single-stage (e.g., SSD and YOLO) framework. However, due to the uncertainty object sizes, a single feature scale cannot meet requirements of the high-accuracy recognition performance. To this end, methods (e.g., SSD and FFP) based on the in-network feature pyramid are proposed and achieve satisfactory results effectively and efficiently. The unified principle behind these methods is to assign region of interest for each object of different size with the appropriate contextual information and enable these objects to be recognized in different feature layers.<br>

## The overall architecture of CFP

![An illustration of the overall architecture](https://github.com/QY1994-0919/CFP-master/assets/overall.png)<br>

## Qualitative results of object detection

![Qualitative results](https://github.com/QY1994-0919/CFP-master/assets/results.png)<br>

# Benchmark<br>
---------
## Standard Models<br>
 Here, we only present the weight file of CEP model experiment with YOLOX as the baseline.<br>
 
| Model | size | mAP(%) | weights |
| :--- | :---: | :---: | ---: |
| CFP-s | 640 | 41.1 | [weight](https://pan.baidu.com/disk/main#/index?category=all&path=%2FCFP-main%2Fweights) | 
| CFP-m | 640 | 46.4 | [weight](https://pan.baidu.com/disk/main#/index?category=all&path=%2FCFP-main%2Fweights) |
| CFP-l | 640 | 49.4 | [weight](https://pan.baidu.com/disk/main#/index?category=all&path=%2FCFP-main%2Fweights) | 

 
# Quick Start<br>
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
                                  cfp-m
                                  cfp-l
                                                                   
## Evaluation: support batch testing for fast evaluation:<br>
                                  
      python -m cfp.tools.eval -n  cfp-s -c cfp_s.pth -b 16 -d 2 --conf 0.001 [--fp16] [--fuse]
                                   cfp-m
                                   cfp-l
                            



# Acknowledgement<br>
 Thanks YOLOX team for the wonderful open source project!

# Citation
If you find CFP useful in your research, please consider citing:<br>
