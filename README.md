# YOLOv3 ROS and PSMNet: Developing a Prototype for the Sidewalk Benchmark Dataset (*SideGuide*)

This repository is a prototype application of ***SideGuide***, a publicly available benchmark dataset for sidewalk navigation.<br/>
SideGuide is a publicly available benchmark designed to assist individuals with sidewalk navigation chellenges.<br/> 
This repository contains the PyTorch code for the "[Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669)" paper (CVPR 2018) by Jia-Ren Chang and Yong-Sheng Chen, as well as the code for the "[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)" paper by Joseph Redmon and Ali Farhadi. More details: http://pjreddie.com/darknet/yolo/

<br/>
본 저장소는 NIA 인도 보행 공공 데이터의 검수용으로 사용된 "Yolo v3 Network" 와 "Pyramid Stereo Matching Network"를 ROS로 통합하여, 실제 인도 보행 환경에서의 객체인식 및 거리추정 성능을 데모로 구현한 것입니다.<br/>
인도 보행 데이터는 공공 데이터 구축을 목적으로 하는 [AI Hub](http://www.aihub.or.kr/) 에서 제공됩니다.<br/>
인도 보행 공공 데이터는 장애인 인도보행의 어려움과 이동권 문제 해결을 위하여 만들어졌습니다.



## Introduction


In ROS, the following connections exist between nodes:

* The Darknet_ros (YOLO) node subscribes to the left image from the ZED camera and publishes information about bounding boxes for detected objects.
* The PSMnet node subscribes to the left and right (stereo) image topics from the ZED camera, estimates disparity, and obtains depth information through the network.
* Finally, by combining the information obtained from the object detection and depth estimation networks, the system displays information for each recognized object, including the type of object and the distance in meters from the camera to the detected object.<br/>

<br>
ROS에서 각 노드는 다음과 같이 연결되어 있습니다. Darknet_ros (YOLO) 노드는 ZED camera로 부터 left image를 subscribe하고  객체인식 결과로 Bounding Box들의 정보를 넘겨줍니다. PSMnet 노드는 ZED camera로 부터 left and right (stereo) image 토픽을 subscribe하고, 네트워크를 통해 disparity를  추정, 깊이정보를 획득합니다. 최종적으로 객체인식 및 깊이추정 네트워크에서 획득한 정보를 결합하여 인식된 객체가 카매라로부터 몇 미터 떨어져 있는지 객체별로 정보를 표시합니다.<br/><br/>



<img align="center" src="https://user-images.githubusercontent.com/25498950/69921866-12429180-14da-11ea-8759-bb23bfb9151f.png">






## <br/>Benchmark Performance

* To obtain the model trained with the SideGuide *Object Detection* / *Instance Segmentation* dataset and find the final AP-50 performance,
  please refer to the "[Download](https://github.com/ytaek-oh/mmdetection/blob/master/docs/BENCHMARK.md)" section.

* For the final performance of the model trained with the *Stereo Matching (disparity estimation)* dataset and to download the trained model,
  please refer to the "[Results on NIA Sidewalk Dataset](https://github.com/parkkibaek/PSMNet_AIHub/blob/master/README.md)" section.


인도보행영상 객체인식/개별 객체분할 데이터셋으로 학습한 모델의 최종 AP-50 성능과 학습된 모델 다운로드 링크는 [Download](https://github.com/ytaek-oh/mmdetection/blob/master/docs/BENCHMARK.md)을 참조하세요. 
거리(깊이) 추정 데이터셋으로 학습한 모델의 최종 성능과 학습된 모델 다운로드 링크는 [Results on NIA Sidewalk Dataset](https://github.com/parkkibaek/PSMNet_AIHub/blob/master/README.md)을 참조하세요. 





## <br/>Benchmark (SideGuide) Download

- #### The sidewalk dataset (SideGuide): [[Download Link]](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=189) for **_Korean access_**
---
:star::star: :fire: ***NEW*** :fire: :star::star:
- #### Please fill out [[this form]](https://docs.google.com/forms/d/e/1FAIpQLScBmoVoj0d-omBOVCHGjhRislXP0TYzRqaUJOmJcqN6ylQcxQ/viewform) to download SideGuide, and we will send you the download link *shortly* after approval.<br/>
  **Will be updated soon!**  
We are currently in the process of creating a new landing page ~~to make the dataset available for download without any national restrictions. 
Once it's ready, we will share the page link here as well. We also aim to have it prepared soon. Thank you for your patience. (10/2023)~~


## <br/>Installation


### Dependencies
Our current implementation is tested on:

- [ZED_SDK v2.8.3](https://www.stereolabs.com/developers/release/#sdkdownloads_anchor)
- [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)
- openCV 2.4.9
- [Python 2.7](https://www.python.org/downloads/)
- [PyTorch 1.0.0](http://pytorch.org)
- torchvision 0.2.0 


### Download weights
Download the two pre-trained weights from our sidewalk dataset--[SideGuide](http://ras.papercept.net/images/temp/IROS/files/1873.pdf).

- YOLOv3 trained on the SideGuide can be downloaded from [here](https://drive.google.com/file/d/1BXQyTGdnB0B5HUffGMeCw5DDCu0S6zRs/view?usp=sharing) and save it to `
catkin_workspace/src/darknet_ros/darknet_ros/yolo_network_config/weights
`
.

- PSMNet trained on the SideGuide can be downloaded from [here](https://github.com/parkkibaek/PSMNet_AIHub/blob/master/README.md) and save it to 
`
catkin_workspace/src/psmnet/src/trained    
`
.

### Build
    cd catkin_workspace/src
    git clone https://github.com/ChelseaGH/sidewalk_prototype_AI_Hub.git
    cd ../
    catkin_make    
**Open a new terminal**

    $ rosalauch zed_wrapper zed.launch
	
&#x279C; ZED CAMERA Connected.

**Open a new terminal** 

    $ roslaunch darknet_ros yolo_v3_sidewalk.launch

&#x279C; Real-time detection results(Bounding Boxes) are shown on the zed camera image. 

**Open a new terminal** 

    $ rosrun psmnet psm_re.py

&#x279C; Detected objects and estimated depths are shown in the terminal window.



### Software

- Ubuntu 16.04.5 LTS (64-bit)
- CUDA Version: 9.0.176

## Contacts
smham@kaist.ac.kr

## License
SideGuide Benchmark:

Copyright (c) 2021 AI Hub

All rights reserved.

The copyright notice must be included when using the data and also for secondary works utilizing this data. This data is built to develop AI technology such as intelligent products, services and can be used for commercial or non-commercial purposes for research and development in various fields.




## Citation

```
@inproceedings{park2020sideguide,
  title={SideGuide: A Large-scale Sidewalk Dataset for Guiding Impaired People},
  author={Park, Kibaek and Oh, Youngtaek and Ham, Soomin and Joo, Kyungdon and Kim, Hyokyoung and Kum, Hyoyoung and Kweon, In So},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={10022--10029},
  year={2020},
  organization={IEEE}
}
```
---

M. Bjelonic
"YOLO ROS: Real-Time Object Detection for ROS",
URL: https://github.com/leggedrobotics/darknet_ros, 2018.
```
@misc{bjelonicYolo2018,
  author = {Marko Bjelonic},
  title = {{YOLO ROS}: Real-Time Object Detection for {ROS}},
  howpublished = {\url{https://github.com/leggedrobotics/darknet_ros}},
  year = {2016--2018},
}
```
```
@inproceedings{chang2018pyramid,
  title={Pyramid Stereo Matching Network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5410--5418},
  year={2018}
}
```
