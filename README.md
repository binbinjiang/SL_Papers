# LATEST Awesome AI Sign Language Papers

Keywords: Sign Language Translation (SLT); Sign Language Recognition (SLR)

## Papers about AI Sign language  [**UPDATING**]

In this repository, we have collected AI sign language (SL) papers for those interested in the field. To facilitate retrieval and viewing, we classify according to different criteria (by time, research content, research institution, etc.). 

If these are useful to you, please **Star** it to support us, and we will update as much as possible.

NOTE: There are overlaps among different categories.


## See Also
Collaborators: 
[Haodong Zhang - Github homepage](https://github.com/0aqz0)

Other useful/recommended collection of AI sign language papers:
[[github]](https://github.com/0aqz0/SLRPapers)


## Table of Content
- **[AI Sign language in Timeline](#2023)**
  - **[2023](#2023)**
  - **[2022](#2022)**
  - **[2021](#2021)**
  - **[2020](#2020)**
  - **[2019](#2019)**
  - **[2018](#2018)**
  - **[Earlier](#Earlier)**

- **[AI Sign language in Types](#slt)**
  - **[Sign Language Translation (SLT)](#slt)**

  - **[Sign Language Recognition (SLR)](#slr)**
    - [CTC-Iteration for Alignment](#ctc-iteration-for-alignment)
    - [CNN+RNN](#cnnrnn)
    - [3D CNN](#3d-cnn)
    - [GCN](#gcn)
    - [Zero-shot](#zero-shot)
    - [Others](#others)
    - [Gesture Recognition](#gesture-recognition) 
    - [Untitle](#untitle)
    - [HMM](#hmm)
  - **[Text2Sign](#text2sign)**
  
  - **[SignText-SignGloss Translation](#text-gloss)**

- **[AI Sign Language in Institutions](#ustc-slr)**
  - **[Koller&Camgoz (英德) SL](#rwth)**
  - **[USTC（中科大）SL](#ustc-slr)**
  - **[XMU (厦门大学) SL](#xmusl)**
  - **[ZJU (浙江大学) SL](#zjusl)**
  - **[THU (清华大学) SL](#thusl)**


- **[SL Survey and Collection](#survey)**


- **[Datasets](#datasets)**

- **[Related Fields](#related-fields)**
  - [Action Recognition](#action-recognition)
  - [Speech Recognition](#speech-recognition)
  - [Video Captioning](#video-captioning)

## Timeline of AI Sign Language

### 2023
- 【CVPR 2023】CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition with Variational Alignment.[[paper]](https://arxiv.org/abs/2303.05725)  ==ZJU&XMU&THU; SLR; Latest==


### 2022
TODO

### 2021
- 【2021 TMM】Spatial-Temporal Multi-Cue Network for Sign Language Recognition and Translation.[[paper]](https://ieeexplore.ieee.org/abstract/document/9354538/)    
  *Hao Zhou, Wengang Zhou, Yun Zhou, Houqiang Li*
- 【ICCV 2021】YouRefIt: Embodied Reference Understanding With Language and Gesture.[[paper]](https://arxiv.org/abs/2109.03413)    
   *Yixin Chen; Qing Li; Deqian Kong; Yik Lun Kei; Song-Chun Zhu; Tao Gao; Yixin Zhu; Siyuan Huang*
- 【ICCV 2021】Speech Drives Templates: Co-Speech Gesture Synthesis With Learned Templates.     
   *Shenhan Qian; Zhi Tu; Yihao Zhi; Wen Liu; Shenghua Gao*
- 【ICCV 2021】Audio2Gestures: Generating Diverse Gestures From Speech Audio With Conditional Variational Autoencoders.    
   *Jing Li; Di Kang; Wenjie Pei; Xuefei Zhe; Ying Zhang; Zhenyu He; Linchao Bao*
- 【ICCV 2021】Aligning Subtitles in Sign Language Videos.[[paper]](https://arxiv.org/abs/2105.02877)     
   *Hannah Bull; Triantafyllos Afouras; Gül Varol; Samuel Albanie; Liliane Momeni; Andrew Zisserman*
- 【ICCV 2021】Mixed SIGNals: Sign Language Production via a Mixture of Motion Primitives.[[paper]](https://arxiv.org/abs/2107.11317)    
   *Ben Saunders; Necati Cihan Camgoz; Richard Bowden*
- 【ICCV 2021】SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition. [[paper]](https://arxiv.org/pdf/2110.05382.pdf)
   *Hezhen Hu; Weichao Zhao; Wengang Zhou; Yuechen Wang; Houqiang Li*
- 【ICCV 2021】Visual Alignment Constraint for Continuous Sign Language Recognition.[[paper]](https://arxiv.org/abs/2104.02330)    [[code]](https://github.com/ycmin95/VAC_CSLR)
   *Yuecong Min, Aiming Hao, Xiujuan Chai, and Xilin Chen*
- 【ICCV 2021】Self-Mutual Distillation Learning for Continuous Sign Language Recognition. [[paper]]()    
  *Aiming Hao, Yuecong Min, and Xilin Chen*
- 【CVPR 2021】Improving Sign Language Translation With Monolingual Data by Sign Back-Translation.[[paper]](https://arxiv.org/abs/2105.12397)    
  *Hao Zhou, Wengang Zhou, Weizhen Qi, Junfu Pu, Houqiang Li*
- 【CVPR 2021】How2Sign: A Large-Scale Multimodal Dataset for Continuous American Sign Language. [[paper]](https://arxiv.org/abs/2008.08143)   
  *Amanda Duarte, Shruti Palaskar, Lucas Ventura, Deepti Ghadiyaram, Kenneth DeHaan, Florian Metze, Jordi Torres, Xavier Giro-i-Nieto*
- 【CVPR 2021】Fingerspelling Detection in American Sign Language. [[paper]](https://arxiv.org/abs/2104.01291)    
  *Bowen Shi, Diane Brentari, Greg Shakhnarovich, Karen Livescu*
- 【CVPR 2021】 Read and Attend: Temporal Localisation in Sign Language Videos.[[paper]](https://arxiv.org/abs/2103.16481)    
  *Gül Varol, Liliane Momeni, Samuel Albanie, Triantafyllos Afouras, Andrew Zisserman*
- 【CVPR 2021】iMiGUE: An Identity-Free Video Dataset for Micro-Gesture Understanding and Emotion Analysis.[[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_iMiGUE_An_Identity-Free_Video_Dataset_for_Micro-Gesture_Understanding_and_Emotion_CVPR_2021_paper.html)    
  *Xin Liu, Henglin Shi, Haoyu Chen, Zitong Yu, Xiaobai Li, Guoying Zhao*
- 【CVPR 2021】Body2Hands: Learning To Infer 3D Hands From Conversational Gesture Body Dynamics. [[paper]](https://arxiv.org/abs/2007.12287)  
  *Evonne Ng, Shiry Ginosar, Trevor Darrell, Hanbyul Joo*
- 【CVPR 2021】Model-Aware Gesture-to-Gesture Translation.[[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_Model-Aware_Gesture-to-Gesture_Translation_CVPR_2021_paper.html)   
  *Hezhen Hu, Weilun Wang, Wengang Zhou, Weichao Zhao, Houqiang Li*
- 【AAAI 2021】Hand-Model-Aware Sign Language Recognition.[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16247)    
  *Hezhen Hu, Wengang Zhou, Houqiang Li*
- 【AAAI 2021】 Regional Attention with Architecture-Rebuilt 3D Network for RGB-D Gesture Recognition.[[paper]](https://arxiv.org/pdf/2102.05348.pdf)    
  *Benjia Zhou, Yunan Li, Jun Wan*
- 【WACV 2021】 Hand Pose Guided 3D Pooling for Word-level Sign Language Recognition.[[paper]](https://openaccess.thecvf.com/content/WACV2021/papers/Hosain_Hand_Pose_Guided_3D_Pooling_for_Word-Level_Sign_Language_Recognition_WACV_2021_paper.pdf)     
  *Al Amin Hosain; Panneer Selvam Santhalingam; Parth Pathak; Huzefa Rangwala; Jana Kosecka*
- 【WACV 2021】Whose hand is this? Person Identification from Egocentric Hand Gestures.[[paper]](https://arxiv.org/abs/2011.08900)      *Satoshi Tsutsui; Yanwei Fu; David Crandall*

### 2020
- 【2020 IJCV】Text2Sign: Towards Sign Language Production Using Neural Machine Translation and Generative Adversarial Networks.[[paper]](<https://link.springer.com/content/pdf/10.1007%2Fs11263-019-01281-2.pdf>)   
  *Stephanie Stoll, Necati Cihan Camgoz, Simon Hadfield, Richard Bowden*
- 【ACM MM 2020】INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.[[paper]](https://dl.acm.org/doi/abs/10.1145/3394171.3413528)    
  *Sridhar, Advaith, Rohith Gandhi Ganesan, Pratyush Kumar, and Mitesh Khapra*
- 【ACM MM 2020】Boosting Continuous Sign Language Recognition via Cross Modality Augmentation.[[paper]](https://arxiv.org/pdf/2010.05264.pdf)    
  *Pu, Junfu, Wengang Zhou, Hezhen Hu, and Houqiang Li*
- 【ACM MM 2020】Recognizing Camera Wearer from Hand Gestures in Egocentric Videos.[[paper]](http://www.cse.iitd.ac.in/~chetan/papers/daksh-mm-2020.pdf)    
  *Thapar, Daksh, Aditya Nigam, and Chetan Arora*
- 【NIPS 2020】TSPNet: Hierarchical Feature Learning via Temporal Semantic Pyramid for Sign Language Translation. [[paper]](https://proceedings.neurips.cc/paper/2020/file/8c00dee24c9878fea090ed070b44f1ab-Paper.pdf)    
  *Li, Dongxu, Chenchen Xu, Xin Yu, Kaihao Zhang, Benjamin Swift, Hanna Suominen, and Hongdong Li*
- 【FG 2020】Feature Selection for Zero-Shot Gesture Recognition. [[paper]](https://www.computer.org/csdl/proceedings-article/fg/2020/307900a309/1kecI8r1WXC)    
  *Naveen Madapana, Juan Wachs*
- 【FG 2020】Image-Based Pose Representation for Action Recognition and Hand Gesture Recognition. [[paper]]()    
  *Zeyi Lin, Wei Zhang, Xiaoming Deng, Cuixia Ma, Hongan Wang*
- 【FG 2020】Neural Sign Language Translation by Learning Tokenization. [[paper]](https://arxiv.org/pdf/2002.00479.pdf)     
   *Orbay, Alptekin, and Lale Akarun* 
- 【FG 2020】Sign Language Recognition in Virtual Reality. [[paper]](https://www.computer.org/csdl/proceedings-article/fg/2020/307900a185/1kecI0aXAje)     
  *Jacob Schioppo, Zachary Meyer, Diego Fabiano, Shaun Canavan*
- 【FG 2020】SILFA: Sign Language Facial Action Database for the Development of Assistive Technologies for the Deaf.[[paper]](https://www.computer.org/csdl/proceedings-article/fg/2020/307900a382/1kecIdR70Y0)    
  *Emely Pujólli da Silva, Paula Dornhofer Paro Costa, Kate Mamhy Oliveira Kumada, José Mario De Martino*
- 【FG 2020】FineHand: Learning Hand Shapes for American Sign Language Recognition. [[paper]](https://arxiv.org/pdf/2003.08753.pdf)    
  *Al Amin Hosain, Panneer Selvam Santhalingam, Parth Pathak, Huzefa Rangwala, Jana Košecká*
- 【FG 2020】Introduction and Analysis of an Event-Based Sign Language Dataset [[paper]](https://www.computer.org/csdl/proceedings-article/fg/2020/307900a441/1kecIyj5b0c)    
  *Ajay Vasudevan, Pablo Negri, Bernabe Linares-Barranco, Teresa Serrano-Gotarredona*
- 【FG 2020】Towards a Visual Sign Language Dataset for Home Care Services. [[paper]](https://www.computer.org/csdl/proceedings-article/fg/2020/307900a622/1kecIMIW1Bm)    
  *D. Kosmopoulos, I. Oikonomidis, C. Constantinopoulos, N. Arvanitis, K. Antzakas, A. Bifis, G. Lydakis, A. Roussos, A. Argyros*
- 【ECCV 2020】 [SLRTP 2020](https://slrtp.com/) Sign language recognition, translation & production. [[Accepted papers]](SLRTP_acccepted_list.md)
- 【ECCV 2020】BSL-1K: Scaling up co-articulated sign language recognition using mouthing cues.[[paper]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560035.pdf)    
  *Samuel Albanie, Gül Varol, Liliane Momeni, Triantafyllos Afouras, Joon Son Chung, Neil Fox, Andrew Zisserman*
- 【ECCV 2020】Progressive Transformers for End-to-End Sign Language Production.[[paper](https://arxiv.org/pdf/2004.14874.pdf)] [[code](https://github.com/BenSaunders27/ProgressiveTransformersSLP)]   
  *Ben Saunders, Necati Cihan Camgoz, and Richard Bowden*
- 【ECCV 2020】Stochastic Fine-grained Labeling of Multi-state Sign Glosses for Continuous Sign Language Recognition.[[paper]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610171.pdf)   
  *Niu Zhe, Brian Mak*
- 【ECCV 2020】Fully Convolutional Networks for Continuous Sign Language Recognition.[[paper](https://arxiv.org/pdf/2007.12402v1.pdf)]   
  *Ka Leong Cheng, Zhaoyang Yang, Qifeng Chen, and Yu-Wing Tai*
- 【ECCV 2020】 Collaborative Learning of Gesture Recognition and 3D Hand Pose Estimation with Multi-Order Feature Analysis.[[paper]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480766.pdf)    
  *Yang, Siyuan, Jun Liu, Shijian Lu, Meng Hwa Er, and Alex C. Kot.*
- 【ECCV 2020】Style Transfer for Co-Speech Gesture Animation: A Multi-Speaker Conditional-Mixture Approach. [[paper]](https://arxiv.org/pdf/2007.12553.pdf)    
  *Chaitanya Ahuja, Dong Won Lee, Yukiko I. Nakano, and Louis-Philippe Morency*
- 【ECCV 2020】Towards Efficient Coarse-to-Fine Networks for Action and Gesture Recognition.[[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750035.pdf)     
  *Quader, Niamul, Juwei Lu, Peng Dai, and Wei Li*
- 【CVPR 2020】Decoupled Representation Learning for Skeleton-Based Gesture Recognition.[[paper]](<http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Decoupled_Representation_Learning_for_Skeleton-Based_Gesture_Recognition_CVPR_2020_paper.pdf>)  
  *Jianbo Liu, Yongcheng Liu, Ying Wang, Véronique Prinet, Shiming Xiang, Chunhong Pan*
- 【CVPR 2020】An Efficient PointLSTM for Point Clouds Based Gesture Recognition.[[paper]](<http://openaccess.thecvf.com/content_CVPR_2020/papers/Min_An_Efficient_PointLSTM_for_Point_Clouds_Based_Gesture_Recognition_CVPR_2020_paper.pdf>)  
  *Yuecong Min, Yanxiao Zhang, Xiujuan Chai, Xilin Chen*
- 【CVPR 2020】Music Gesture for Visual Sound Separation. [[paper]](<https://arxiv.org/pdf/2004.09476.pdf>)  
  *Chuang Gan, Deng Huang, Hang Zhao, Joshua B. Tenenbaum, Antonio Torralba*	
- 【CVPR 2020】Transferring Cross-Domain Knowledge for Video Sign Language Recognition.[[paper]](<https://arxiv.org/pdf/2003.03703.pdf>)  
  *Dongxu Li, Xin Yu, Chenchen Xu, Lars Petersson, Hongdong Li*	
- 【CVPR 2020] Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation.[[paper]](<https://arxiv.org/pdf/2003.13830.pdf>)   
  *Necati Cihan Camgöz, Oscar Koller, Simon Hadfield, Richard Bowden*	
- 【AAAI 2020】Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition. [[paper]](<https://arxiv.org/pdf/2002.03187.pdf>)  
  *Hao Zhou, Wengang Zhou, Yun Zhou, Houqiang Li* 
- 【WACV 2020】Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison.  [[paper]](<http://openaccess.thecvf.com/content_WACV_2020/papers/Li_Word-level_Deep_Sign_Language_Recognition_from_Video_A_New_Large-scale_WACV_2020_paper.pdf>)  
  *Dongxu Li, Cristian Rodriguez, Xin Yu, Hongdong Li*
- 【WACV 2020】Neural Sign Language Synthesis: Words Are Our Glosses. [[paper]](<http://openaccess.thecvf.com/content_WACV_2020/papers/Zelinka_Neural_Sign_Language_Synthesis_Words_Are_Our_Glosses_WACV_2020_paper.pdf>)  
  *Jan Zelinka, Jakub Kanis*


## SLT
1. **Multi-channel Transformers for Multi-articulatory Sign Language Translation** `arxiv2020` [*paper*](https://arxiv.org/pdf/2009.00299.pdf) *code*
1. **Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation** `CVPR2020` [*paper*](https://arxiv.org/pdf/2003.13830.pdf) [*code*](https://github.com/neccam/slt) ==德国==
2. **Sign Language Translation with Transformers** `ArXiv2020` [*paper*](https://arxiv.org/pdf/2004.00588.pdf) [*code*](https://github.com/kayoyin/transformer-slt) 
3. **Neural Sign Language Translation by Learning Tokenization** `Arxiv2020` [*paper*](https://arxiv.org/pdf/2002.00479.pdf) *code* 
4. **Neural Sign Language Translation based on Human Keypoint Estimation** `Arxiv2018` [*paper*](https://arxiv.org/pdf/1811.11436.pdf) *code* 
5. **Neural Sign Language Translation** `CVPR2018` [*paper*](http://openaccess.thecvf.com/content_cvpr_2018/papers/Camgoz_Neural_Sign_Language_CVPR_2018_paper.pdf) [*code*](https://github.com/neccam/nslt)


## SLR

### CTC-Iteration for Alignment
* mainly CNN+RNN*

1. **Stochastic Fine-grained Labeling of Multi-state Sign Glosses for Continuous Sign Language Recognition** `ECCV2020` [*paper*](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610171.pdf) [*code*](https://github.com/zheniu/stochastic-cslr) 
4. **Fully Convolutional Networks for Continuous Sign Language Recognition** `ECCV2020` [*paper*](https://arxiv.org/pdf/2007.12402v1.pdf) *code* 
1. **Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition** `AAAI2020` [*paper*](https://arxiv.org/pdf/2002.03187.pdf) *code* ==中科大&对齐==
2. **Dense Temporal Convolution Network for Sign Language Translation** `IJCAI2019` [*paper*](https://www.ijcai.org/Proceedings/2019/0105.pdf) *code* ==合工大==
3. **A Deep Neural Framework for Continuous Sign
Language Recognition by Iterative Training** `IEEE TRANSACTIONS ON MULTIMEDIA 2019` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8598757&tag=1) *code* ==中科大&对齐==
4. **Iterative Alignment Network for Continuous Sign Language** `CVPR2019` [*paper*](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pu_Iterative_Alignment_Network_for_Continuous_Sign_Language_Recognition_CVPR_2019_paper.pdf) *code* ==中科大&对齐==
5. **Weakly Supervised Learning with Multi-Stream CNN-LSTM-HMMs to Discover Sequential Parallelism in Sign Language Videos** `TRAMI2019` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8691602) [*code 非官方caffe*](https://github.com/huerlima/Re-Sign-Re-Aligned-End-to-End-Sequence-Modelling-with-Deep-Recurrent-CNN-HMMs) ==Koller&开源&对齐==
6. **SF-Net: Structured Feature Network for Continuous Sign Language Recognition** `ArXiv2019` [*paper*](https://arxiv.org/pdf/1908.01341.pdf) *code*
7. **Dilated Convolutional Network with Iterative Optimization for Continuous Sign Language Recognition** `IJCAI2018` [*Paper*](https://www.ijcai.org/Proceedings/2018/0123.pdf) [*code*](https://github.com/ustc-slr/DilatedSLR) ==开源&中科大==
8. **Re-Sign: Re-Aligned End-to-End Sequence Modelling with Deep Recurrent CNN-HMMs** `CVPR2017` [*paper*](https://www-i6.informatik.rwth-aachen.de/publications/download/1031/KollerOscarZargaranSepehrNeyHermann--Re-SignRe-AlignedEnd-to-EndSequenceModellingwithDeepRecurrentCNN-HMMs--2017.pdf) [*code 非官方caffe*](https://github.com/huerlima/Re-Sign-Re-Aligned-End-to-End-Sequence-Modelling-with-Deep-Recurrent-CNN-HMMs) ==Koller&开源&对齐==
9. **Recurrent Convolutional Neural Networks for Continuous Sign Language Recognition by Staged Optimization** `CVPR2017` [*paper*](http://openaccess.thecvf.com/content_cvpr_2017/papers/Cui_Recurrent_Convolutional_Neural_CVPR_2017_paper.pdf) *code* ==对齐==



### GCN
*GCN for SLR*

1. **Spatial-Temporal Graph Convolutional Networks for Sign Language Recognition** `ICANN2019` [*paper*](https://arxiv.org/pdf/1901.11164.pdf) [*code*](https://github.com/amorim-cleison/st-gcn-sl)
2. **开源项目(浙大某研究生),包括了GCN for SLR代码** *paper* [*code*](https://github.com/0aqz0/SLR)


#### **GCN-related Work*
*GCN for Action Recognition:*
1. Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition [*paper*]( https://arxiv.org/pdf/1801.07455.pdf) [*code*](https://github.com/yysijie/st-gcn)

*GCN & Zero-shot:*
2. Zero-shot Recognition via Semantic Embeddings and Knowledge Graphs [*paper*](https://arxiv.org/pdf/1803.08035.pdf) [*code*](https://github.com/JudyYe/zero-shot-gcn) ==GCN&zero-shot==



### Zero-shot
1.**Zero-Shot Sign Language Recognition: Can Textual Data Uncover Sign Languages?** `BMVC2019` [*paper*](https://arxiv.org/pdf/1907.10292.pdf) [*code 非官方 浙大某学生*](https://github.com/lwj2018/islr-few-shot)

#### **低资源相关论文*
1. **Fingerspelling recognition in the wild with iterative visual attention** `CVPR2018` [*paper*](https://arxiv.org/pdf/1908.10546.pdf) *code*
2. [综述](https://github.com/sbharadwajj/awesome-zero-shot-learning)


### Others

#### Gesture Recognition
1. **Fingerspelling recognition in the wild with iterative visual attention** `ICCV2019` [*paper*](https://arxiv.org/pdf/1908.10546.pdf) *code*
7. **Attention in Convolutional LSTM for Gesture Recognition** `NIPS2018` [*paper*](http://papers.nips.cc/paper/7465-attention-in-convolutional-lstm-for-gesture-recognition.pdf) [*code*](https://github.com/GuangmingZhu/AttentionConvLSTM)  ==开源==
8. **SubUNets: End-to-End Hand Shape and Continuous Sign Language Recognition** `ICCV2017` [*paper*](http://openaccess.thecvf.com/content_ICCV_2017/papers/Camgoz_SubUNets_End-To-End_Hand_ICCV_2017_paper.pdf) [*code*](https://github.com/neccam/SubUNets) ==开源&Camgoz&CNN+RNN==
2. **Learning Spatiotemporal Features Using 3DCNN and Convolutional LSTM for Gesture Recognition** `ICCV2017` [*paper*](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w44/Zhang_Learning_Spatiotemporal_Features_ICCV_2017_paper.pdf) [*code*](https://github.com/GuangmingZhu/Conv3D_BICLSTM) ==开源&CNN+RNN==
2. **Using Convolutional 3D Neural Networks for User-independent continuous gesture recognition** `ICPR2016` [*paper*](http://personal.ee.surrey.ac.uk/Personal/S.Hadfield/papers/camgoz2016icprw.pdf) *code*
3. **Online Detection and Classification of Dynamic Hand Gestures with Recurrent 3D Convolutional Neural Networks** `CVPR2016` [*paper*](https://research.nvidia.com/sites/default/files/pubs/2016-06_Online-Detection-and/NVIDIA_R3DCNN_cvpr2016.pdf) *code*
4. **Hand Gesture Recognition with 3D Convolutional Neural Networks** `CVPRW2015` [*paper*](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W15/papers/Molchanov_Hand_Gesture_Recognition_2015_CVPR_paper.pdf) *code*
1. **Sign Language Recognition Analysis using Multimodal Data** `DSAA2019` [*paper*](https://arxiv.org/pdf/1909.11232.pdf) *code*


#### Untitle

1. **BSL-1K: Scaling up co-articulated sign language recognition using mouthing cues** `ECCV2020` [*paper*](https://arxiv.org/pdf/2007.12131.pdf) *code*
1. **Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison** `WACV2020` [*paper*](https://arxiv.org/pdf/1910.11006.pdf) [*code*](https://github.com/dxli94/WLASL) 
2. **Dynamic Sign Language Recognition Based on Video Sequence With BLSTM-3D Residual Networks** `ACCESS2019` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8667292) *code*
3. **Temporal Accumulative Features for Sign Language Recognition** `ICCV2019`  [*paper*](https://arxiv.org/pdf/2004.01225.pdf) *code*
4. **Thai Sign Language Recognition Using 3D Convolutional Neural Networks** `ICCCM2019` [*paper*](https://dl.acm.org/doi/pdf/10.1145/3348445.3348452?download=true) *code*
4. **Human-like sign-language learning method using deep learning** `ETRI2018` [*paper*](https://onlinelibrary.wiley.com/doi/pdf/10.4218/etrij.2018-0066) *code*

6. **Deep Hand: How to Train a CNN on 1 Million Hand Images When Your Data is Continuous and Weakly Labelled** `CVPR2016` [*paper*](https://www-i6.informatik.rwth-aachen.de/publications/download/1000/KollerOscarNeyHermannBowdenRichard--DeepHHowtoTrainaCNNon1MillionHImagesWhenYourDataIsContinuousWeaklyLabelled--2016.pdf) *code*
7. **Deep Sign: Hybrid CNN-HMM for Continuous Sign Language Recognition** `BMVC2016` [*paper*](https://pdfs.semanticscholar.org/7b2f/db4a2f79a638ad6c5328cd2860b63fdfc100.pdf) *code*
8. **SIGN LANGUAGE RECOGNITION WITH LONG SHORT-TERM MEMORY** `ICIP2016` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7532884) *code*
9. **Iterative Reference Driven Metric Learning for Signer Independent Isolated Sign Language Recognition.** `ECCV2016` [*paper*](http://vipl.ict.ac.cn/uploadfile/upload/2018112115134267.pdf) *code*
10. **Automatic Alignment of HamNoSys Subunits for Continuous Sign Language Recognition** `LREC2016` [*paper*](https://pdfs.semanticscholar.org/f6a3/c3ab709eebc91f9639fe6d26b7736c3115b2.pdf?_ga=2.172107747.1969091582.1582720499-418088591.1578543327) *code*
11. **Sign Language Recognition using 3D convolutional neural networks** `ICME2015` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7177428) *code*

11. **Curve Matching from the View of Manifold for Sign Language Recognition** `ACCV2014` [*paper*](http://whdeng.cn/FSLCV14/pdffiles/w12-o7.pdf) *code*
12. **Sign Language Recognition and Translation with Kinect** `AFGR2013` [*paper*](https://pdfs.semanticscholar.org/0450/ecef50fd1f532fe115c5d32c7c3ebed6fd80.pdf?_ga=2.205538387.1969091582.1582720499-418088591.1578543327) *code*
13. **Large-scale Learning of Sign Language by Watching TV (Using Cooccurrences).** `BMVC2013` [*paper*](http://www.robots.ox.ac.uk:5000/~vgg/publications/2013/Pfister13/pfister13.pdf) *code*
14. **Sign Language Recognition using Sequential Pattern Trees** `CVPR2012` [*paper*](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.261.3830&rep=rep1&type=pdf) *code*
15. **Sign language recognition using sub-units** `JMLR2012` [*paper*](http://www.jmlr.org/papers/volume13/cooper12a/cooper12a.pdf) *code*
16. **American Sign Language word recognition with a sensory glove using artificial neural networks** `Eng.Appl.Artif.Intell.2011` [*paper*](https://pdf.sciencedirectassets.com/271095/1-s2.0-S0952197611X00076/1-s2.0-S0952197611001230/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFQaCXVzLWVhc3QtMSJIMEYCIQCDZxnJOHB6ynrfo%2B0eKqcxZx3UyY6n1LVRR2QvdupwOAIhAOfZShxaTYLFHn%2F72DKrv9t%2BmLGlEJRfMa8lgTOzU4PaKr0DCM3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQAhoMMDU5MDAzNTQ2ODY1IgyZF8SDS6TiuKrS8eEqkQOVoR%2BTiMf07fTZ5J84%2FjQZYJwK%2F2xGNhjzM7v5ISMQI%2FCDEm2qvcsc0YpaTk%2B2BbEwYYO7yVzPyexMkyl%2FklBtl4sxrAa9Dfc3wJs%2FbRRSh6JbbF4kX85PQ8%2BnrnZ7at9281HfUYhCVk8Sct75U2C4LdfBKUEvk0R0cf%2FkIzfi9mXF1IQ9oy1fpLG6q49kH28Arx5g3FlNKs3IjR7oIZy4pYkcdNnjd4kJxPtExRWmlFi1nITofQGITzNdsEtDGQIdk7M2HsbHoSlLwgqGirrFCLRnkUJ8XY9HoD4%2Fe%2FqG4q7knK4oshtBBQy3CNVp6X%2FpnkQ%2B2dcBxs9xoWXalwuEsdv8PDtP1MACScDfNIsDoRUA6vjdNzPW3aSPBDjwA6lkEE5StCKn55TXIQ6Zd6Y7IxfjUnMgOFWAQUa%2FvNXoFjzCjwMPZDO4zOsWuW32h3W2%2F3Rf2SSjf%2BBa112eh5%2FNeTV9iddv1TiAvt37jyb7n2xiCipeNYfpVHk1Ocj7367O1QDbwF2FX1byN6ULKJGhCjDSwNrwBTrqAQYivYa%2BOo26NX78i0ddvdNA%2FiTXdcs7U3QUcoG7GB49i8Ufd9mEGbEWZcDu3hYc8dRsq93ETnKKfI64YRfVljn4RfMrWeHghUnxALq448eFfuB5LhIsMDVwyaUE%2BUdD40W1hypCdpcWmgAZYuYJNcCCSPYlmQYd5fn9bFcGZD235aXKzCUntJiWClVCblsgKdzmvupUKOBz3xrTdMbn%2FTjktLpjGO9WnJJ1YGJlsZxQELb4snrTQU8kOUeonYg6rjduZkrE9TO%2Fw898abdJr60xIrx9sVLuWK3vbfDezkJv5xVXrC6CVYQQAw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200109T045503Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYZOOW47M%2F20200109%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f24d01b3b6674ba20289c504c727ffe5cccfec13d342832f6b323c81ec689f1f&hash=4b882e75b9a4d7296ee682513ba3630c7d34190549527aec5c1644be54b5b32b&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0952197611001230&tid=spdf-306582c7-3593-4d10-a68f-e6570a446d78&sid=b06c3dbe4dd5d04d742bcc8-a5a302ee0a5agxrqa&type=client) *code*
17. **Learning sign language by watching TV (using weakly aligned subtitles)** `CVPR2009` [*paper*](https://www.robots.ox.ac.uk/~vgg/publications/2009/Buehler09/buehler09.pdf) *code*


### G2T

1. **Translation of sign language glosses to text using sequence-to-sequence attention models.** `SITIS2019` [*paper*](https://www.researchgate.net/publication/340689060_Translation_of_Sign_Language_Glosses_to_Text_Using_Sequence-to-Sequence_Attention_Models) *code* ==ASLG-PC12数据集==


### Dataset Paper

1. **English-ASL Gloss Parallel Corpus 2012: ASLG-PC12, The Second Release** `ICTA2013` [*paper*](https://www.researchgate.net/profile/Achraf_Othman/publication/227339312_English-ASL_Gloss_Parallel_Corpus_2012_ASLG-PC12/links/09e414fe0451f44d14000000.pdf) *code*


### HMM

1. **Online Early-Late Fusion Based on Adaptive HMM for Sign Language Recognition** `TOMM2017` [*paper*](https://dl.acm.org/doi/pdf/10.1145/3152121?download=true) *code*
2. **Chinese sign language recognition with adaptive HMM** `ICME2016` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7552950) *code*
3. **Sign language recognition based on adaptive HMMS with data augmentation** `ICIP2016` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7532885) *code*
4. **Continuous sign language recognition using level building based on fast hidden Markov model** `Pattern Recognit.Lett.2016` [*paper*](https://pdf.sciencedirectassets.com/271524/1-s2.0-S0167865516X00086/1-s2.0-S0167865516300344/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEAaCXVzLWVhc3QtMSJHMEUCIQD2Y%2BxR5o8TZTHS2281Y35EvaUofwEtvu1ZEd9IzltFCAIgdTNMHwp2zGmEjq7mpzo2ewVyzh2Mn5zTTR0H29nDY74qvQMIuP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgwwNTkwMDM1NDY4NjUiDEar6BbfeNCgeErvDSqRA1dz9hH9CR6nilnRKmUSYaZl7Fk4rSmfywj7DUDvdCVEGeo%2BoVVLCBrZl2SHby8RITLRTXhklXfrgnn1ek%2Fgynu9sb4H6eBzKJbqET6t2JbONb7iT8NEkqFX1GUVuh5xAvdFNux8v5EwC6j7V9wXbU2WAIuuZJKUN1rK4JJTIo6ww7%2FwjPPn5XGCVQfc1Wp6Oz6j1KScAnPgZzJ1h8MWarW5SqoeGxO0kPH1y1%2BUxOESHFvsZPfB5nmRuqW4rON9m9kb%2BgD0PY0MSn%2BBo19xekFkhc%2FowTkHyU7kw7tPfSa0XRQuTjEXah0m80fzKmgGChHW8u4GPUFqCV1aunbolvf%2BFTJNm2gvUq5W4a%2Fyvhr5%2Fd%2FRQxWCcZCXuMRWIRm0IGEc6Ug8fyPgn2CplAtVYMTmQsqUJmaXLxQD6HmQ6IJiY5g7hCOvXjC%2F6jgr2L4SdvOTohIlzYB5LLek3rm%2BCB4JgoWmGuT74Iv%2FV4vxjexmTE1WegVcyQ%2BsIF%2B%2Bj7bXO%2FcWymf5NsMoBUt35jWu3wTuMJSE1vAFOusBtxzItLiNNvLLwB%2FR1a8vxlW3a10tX8Tg1oVxQDAy1xT%2BJ0BPrQuB3Z555vyScCn1IY0%2FsQGoh%2BkOAKlilzLDRxdwA3T5lbWfPaCb2cK31pwbFv1XkmxwJ3K2w5kziZ0J6n8aiwespNME9qUrzvmZ3wB8N3fvicmYs4nKCLHSyDCScXOX5stLaY069z5G6sMrnr05wATDlolASJtMXj5u5m2b381hnpmVPbT%2Ft7lueZoEGTa%2FCO%2BRUx0tq5J4rAaVOtC%2Fokh8yuMVasm4AZR1GgSlGl97UEKxdo0JaZ6c%2Bx%2Bh7BKeWNJEyNjUqw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20200108T074501Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYWB765V7N%2F20200108%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=8f578a751f8ba78bf4bdf010c4d22d373725e6b6dc3cd45fbfc85838b1450d2e&hash=a4caf615747056ee196c29b41bf9170d9ea0f54106703ae52f095ab03f51f717&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0167865516300344&tid=spdf-dd403efa-8ad7-4a95-9e7c-99bdb33484a3&sid=2e1bb56239e50144b748444812c269de5761gxrqa&type=client) *code*
5. **Sign Transition Modeling and a Scalable Solution to Continuous Sign Language Recognition for Real-World Applications** `TACCESS2016` [*paper*](https://dl.acm.org/doi/pdf/10.1145/2850421?download=true) *code*
6. **A Threshold-based HMM-DTW Approach for Continuous Sign Language Recognition** `ICIMCS2014` [*paper*](https://dl.acm.org/doi/pdf/10.1145/2632856.2632931?download=true) *code*
7. **Improving Continuous Sign Language Recognition: Speech Recognition Techniques and System Design** `SLPAT2013` [*paper*](https://pdfs.semanticscholar.org/91e4/220449ea1d7ed2b49c916dd89af850c69b26.pdf) *code*
8. **Using Viseme Recognition to Improve a Sign Language Translation System** `IWSLT2013` [*paper*](https://pdfs.semanticscholar.org/e567/428f531e973a4544f1884d9d7e7aa59953a6.pdf?_ga=2.235954241.1969091582.1582720499-418088591.1578543327) *code*
9. **Advances in phonetics-based sub-unit modeling for transcription alignment and sign language recognition** `CVPRW2011` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5981681&tag=1) *code*
10. **Speech Recognition Techniques for a Sign Language Recognition System** `INTERSPEECH2007` [*paper*](https://www-i6.informatik.rwth-aachen.de/publications/download/154/DreuwPhilippeRybachDavidDeselaersThomasZahediMortezaNeyHermann--SpeechRecognitionTechniquesforaSignLanguageRecognitionSystem--2007.pdf) *code*
11. **Large-Vocabulary Continuous Sign Language Recognition Based on Transition-Movement Models** `TSMC2007` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4032919) *code*
12. **Real-time American sign language recognition using desk and wearable computer based video** `TPAMI1998`  [*paper*](http://luthuli.cs.uiuc.edu/~daf/courses/Signals%20AI/Papers/HMMs/00735811.pdf) *code*



## Text2Sign

1. **Text2Sign: Towards Sign Language Production Using Neural Machine Translation and Generative Adversarial Networks** `IJCV2020` [*paper*](https://link.springer.com/content/pdf/10.1007%2Fs11263-019-01281-2.pdf) *code*
2. **Progressive Transformers for End-to-End Sign Language Production** `ECCV2020` [*paper*](https://arxiv.org/pdf/2004.14874.pdf) *code*




## USTC SLR

##### Conference papers:
1. **Boosting Continuous Sign Language Recognition via Cross Modality Augmentation** `ACMMM2020` 
1. **Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition** `AAAI2020` [*paper*](https://arxiv.org/pdf/2002.03187.pdf) *code* 
1. **Deep Grammatical Multi-classifier for Continuous Sign Language Recognition** `BigMM2019` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8919458) *code*
1. **Continuous Sign Language Recognition via Reinforcement Learning** `ICIP2019` *paper* *code* ==#TODO==
1. **Iterative Alignment Network for Continuous Sign Language** `CVPR2019` [*paper*](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pu_Iterative_Alignment_Network_for_Continuous_Sign_Language_Recognition_CVPR_2019_paper.pdf) *code* 
1. **Dynamic Pseudo Label Decoding for Continuous Sign Language Recognition** `ICME 2019`. [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8784863) *code* 
1.  **Dilated Convolutional Network with Iterative Optimization for Continuous Sign Language Recognition** `IJCAI2018` [*Paper*](https://www.ijcai.org/Proceedings/2018/0123.pdf) [*code*](https://github.com/ustc-slr/DilatedSLR) 
1. **Video-based sign language recognition without temporal segmentation** `AAAI2018` [*paper*](https://arxiv.org/pdf/1801.10111.pdf) *code*  ==CNN+RNN==
1. **Hierarchical LSTM for Sign Language Translation** `AAAI2018` [*paper*](https://pdfs.semanticscholar.org/d44c/20c48e764a546d00b9155a56b171b0dc04bc.pdf) *code* ==CNN+RNN==
1. **Connectionist Temporal Fusion for Sign Language Translation** `ACMMM018` [*paper*](https://dl.acm.org/doi/pdf/10.1145/3240508.3240671) *code*  ==CNN+RNN==
1. **Chinese Sign Language Recognition with Adaptive HMM** `ICME2016` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7552950) *code*
1. **Sign Language Recognition with Multi-modal Features** `PCM2016`
1. **Sign Language Recognition with Long Short Term Memory** `ICIP2016`
1. **Sign Language Recognition based on Adaptive HMMs with Data Augmentation** `ICIP2016`
1. **Sign Language Recognition Based on Trajectory Modeling with HMMs** `MMM2016`
1. **Sign Language Recognition using Real-Sense** `ChinaSIP2015`
1. **A New System forChinese Sign Language Recognition** `ChinaSIP2015`
1. **Sign language recognition using 3D convolutional neural networks** `ICME2015`
1. **A Threshold-based HMM-DTW Approach for Continuous Sign Language Recognition** `ICIMCS2014`

##### Journal papers:
1. **Semantic Boundary Detection with Reinforcement Learning for Continuous Sign Language Recognition** `TCSVT2020`  ==#TODO 没找到该文章 应该还没公布==
2. **Attention based 3D-CNNs for Large-Vocabulary Sign Language Recognition** `TCSVT2018` *paper* *code* ==#TODO==



## RWTH
*英国Oscar Koller、Camgoz等人工作的专题（与前面有重复部分） 
1. **Multi-channel Transformers for Multi-articulatory Sign Language Translation** `arxiv2020` [*paper*](https://arxiv.org/pdf/2009.00299.pdf) *code*
1. **Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation** `CVPR2020` [*paper*](https://arxiv.org/pdf/2003.13830.pdf) [*code*](https://github.com/neccam/slt) ==德国==
2. **Sign Language Translation with Transformers** `ArXiv2020` [*paper*](https://arxiv.org/pdf/2004.00588.pdf) [*code*](https://github.com/kayoyin/transformer-slt)
5. **Neural Sign Language Translation** `CVPR2018` [*paper*](http://openaccess.thecvf.com/content_cvpr_2018/papers/Camgoz_Neural_Sign_Language_CVPR_2018_paper.pdf) [*code*](https://github.com/neccam/nslt)
5. **Weakly Supervised Learning with Multi-Stream CNN-LSTM-HMMs to Discover Sequential Parallelism in Sign Language Videos** `TRAMI2019` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8691602) [*code 非官方caffe*](https://github.com/huerlima/Re-Sign-Re-Aligned-End-to-End-Sequence-Modelling-with-Deep-Recurrent-CNN-HMMs) ==Koller&开源&对齐==
8. **Re-Sign: Re-Aligned End-to-End Sequence Modelling with Deep Recurrent CNN-HMMs** `CVPR2017` [*paper*](https://www-i6.informatik.rwth-aachen.de/publications/download/1031/KollerOscarZargaranSepehrNeyHermann--Re-SignRe-AlignedEnd-to-EndSequenceModellingwithDeepRecurrentCNN-HMMs--2017.pdf) [*code 非官方caffe*](https://github.com/huerlima/Re-Sign-Re-Aligned-End-to-End-Sequence-Modelling-with-Deep-Recurrent-CNN-HMMs) ==Koller&开源&对齐==


### Neural SLT

1. **Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation** `CVPR2020` [*paper*](https://arxiv.org/pdf/2003.13830.pdf) [*code*](https://github.com/neccam/slt) ==德国==
2. **Neural Sign Language Translation** `CVPR2018` [*paper*](http://openaccess.thecvf.com/content_cvpr_2018/papers/Camgoz_Neural_Sign_Language_CVPR_2018_paper.pdf) [*code*](https://github.com/neccam/nslt)


### SLR  ---TODO
1. **Sign Language Recognition, Generation, and Translation: An Interdisciplinary Perspective** `ASSETS2019` [*paper*](https://dl.acm.org/doi/10.1145/3308561.3353774) *code*

1. **Deep Sign: Enabling Robust Statistical Continuous Sign Language Recognition via Hybrid CNN-HMMs** `IJCV2018` [*paper*](https://dl.acm.org/doi/10.1007/s11263-018-1121-3) *code*

1. **Deep Learning of Mouth Shapes for Sign Language** `ICCVW2015` [*paper*](https://dl.acm.org/doi/10.1109/ICCVW.2015.69) *code*


## XMU SL 
- 【CVPR 2023】CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition with Variational Alignment.[[paper]](https://arxiv.org/abs/2303.05725)  ==ZJU&XMU&THU; SLR; Latest==


## ZJU SL 
- 【CVPR 2023】CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition with Variational Alignment.[[paper]](https://arxiv.org/abs/2303.05725)  ==ZJU&XMU&THU; SLR; Latest==


## THU SL 
- 【CVPR 2023】CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition with Variational Alignment.[[paper]](https://arxiv.org/abs/2303.05725)  ==ZJU&XMU&THU; SLR; Latest==

## Datasets

| Dataset                                                      | Language    | Classes | Samples | Data Type                    | Language Level |
| ------------------------------------------------------------ | ----------- | ------- | ------- | ---------------------------- | :------------- |
| **[ASLG-PC12](https://github.com/kayoyin/transformer-slt/tree/master/data) ->暂时没找到官方链接；used by [paper](https://arxiv.org/pdf/2004.00588.pdf)** | American     | -     | 87,709 | GLOSS&Sentences     | isolated       |
| **[CSL Dataset I](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html)** | Chinese     | 500     | 125,000 | Videos&Depth from Kinect     | isolated       |
| **[CSL Dataset II](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html)** | Chinese     | 100     | 25,000  | Videos&Depth from Kinect     | continuous     |
| **[RWTH-PHOENIX-Weather 2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)** | German      | 1,081   | 6,841   | Videos                       | continuous     |
| [**RWTH-PHOENIX-Weather 2014 T**](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) | German      | 1,066   | 8,257   | Videos                       | continuous     |
| **[ASLLVD](http://www.bu.edu/asllrp/av/dai-asllvd.html)**    | American    | 3,300   | 9,800   | Videos(multiple angles)      | isolated       |
| **[ASLLVD-Skeleton](https://www.cin.ufpe.br/~cca5/asllvd-skeleton/)** | American    | 3,300   | 9,800   | Skeleton                     | isolated       |
| **[SIGNUM](https://www.phonetik.uni-muenchen.de/forschung/Bas/SIGNUM/)** | German      | 450     | 33,210  | Videos                       | continuous     |
| [**DGS Kinect 40**](http://personal.ee.surrey.ac.uk/Personal/H.Cooper/research/papers/Ong_Sign_2012.pdf) | German      | 40      | 3,000   | Videos(multiple angles)      | isolated       |
| [**DEVISIGN-G**](http://vipl.ict.ac.cn/homepage/ksl/data.html) | Chinese     | 36      | 432     | Videos                       | isolated       |
| [**DEVISIGN-D**](http://vipl.ict.ac.cn/homepage/ksl/data.html) | Chinese     | 500     | 6,000   | Videos                       | isolated       |
| [**DEVISIGN-L**](http://vipl.ict.ac.cn/homepage/ksl/data.html) | Chinese     | 2000    | 24,000  | Videos                       | isolated       |
| [**LSA64**](http://facundoq.github.io/unlp/lsa64/)           | Argentinian | 64      | 3,200   | Videos                       | isolated       |
| [**GSL isol.**](https://vcl.iti.gr/dataset/gsl/)             | Greek       | 310     | 40,785  | Videos&Depth from RealSense  | isolated       |
| [**GSL SD**](https://vcl.iti.gr/dataset/gsl/)                | Greek       | 310     | 10,290  | Videos&Depth from RealSense  | continuous     |
| [**GSL SI**](https://vcl.iti.gr/dataset/gsl/)                | Greek       | 310     | 10,290  | Videos&Depth from RealSense  | continuous     |
| [**IIITA -ROBITA**](https://robita.iiita.ac.in/dataset.php)  | Indian      | 23      | 605     | Videos                       | isolated       |
| [**PSL Kinect**](http://vision.kia.prz.edu.pl/dynamickinect.php) | Polish      | 30      | 300     | Videos&Depth from Kinect     | isolated       |
| [**PSL ToF**](http://vision.kia.prz.edu.pl/dynamictof.php)   | Polish      | 84      | 1,680   | Videos&Depth from ToF camera | isolated       |
| [**BUHMAP-DB**](https://www.cmpe.boun.edu.tr/pilab/pilabfiles/databases/buhmap/) | Turkish     | 8       | 440     | Videos                       | isolated       |
| [**LSE-Sign**](http://lse-sign.bcbl.eu/web-busqueda/)        | Spanish     | 2,400   | 2,400   | Videos                       | isolated       |
| [**Purdue RVL-SLLL**](https://engineering.purdue.edu/RVL/Database/ASL/asl-database-front.htm) | American    | 39      | 546     | Videos                       | isolated       |
| [**RWTH-BOSTON-50**](http://www-i6.informatik.rwth-aachen.de/aslr/database-rwth-boston-50.php) | American    | 50      | 483     | Videos(multiple angles)      | isolated       |
| [**RWTH-BOSTON-104**](http://www-i6.informatik.rwth-aachen.de/aslr/database-rwth-boston-104.php) | American    | 104     | 201     | Videos(multiple angles)      | continuous     |
| [**RWTH-BOSTON-400**](http://www-i6.informatik.rwth-aachen.de/~dreuw/database.php) | American    | 400     | 843     | Videos                       | continuous     |
| [**WLASL**](https://dxli94.github.io/WLASL/)                 | American    | 2,000   | 21,083  | Videos                       | isolated       |



## Related Fields

### Action Recognition

1. **DistInit: Learning Video Representations Without a Single Labeled Video** `ICCV2019` [*paper*](https://arxiv.org/pdf/1901.09244.pdf) *code*
2. **SCSampler: Sampling Salient Clips from Video for Efficient Action Recognition** `ICCV2019` [*paper*](http://openaccess.thecvf.com/content_ICCV_2019/papers/Korbar_SCSampler_Sampling_Salient_Clips_From_Video_for_Efficient_Action_Recognition_ICCV_2019_paper.pdf) *code*
3. **Reasoning About Human-Object Interactions Through Dual Attention Networks** `ICCV2019` [*paper*](https://arxiv.org/pdf/1909.04743) *code*
4. **SlowFast Networks for Video Recognition** `ICCV2019` [*paper*](http://openaccess.thecvf.com/content_ICCV_2019/papers/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.pdf) [*code*](https://github.com/facebookresearch/SlowFast)
5. **Video Classification with Channel-Separated Convolutional Networks** `ICCV2019` [*paper*](https://arxiv.org/pdf/1904.02811) [*code*](https://github.com/facebookresearch/VMZ)
6. **BMN: Boundary-Matching Network for Temporal Action Proposal Generation** `ICCV2019` [*paper*](https://arxiv.org/pdf/1907.09702) [*code*](https://github.com/JJBOY/BMN-Boundary-Matching-Network)
7. **DynamoNet: Dynamic Action and Motion Network** `ICCV2019` [*paper*](https://arxiv.org/pdf/1904.11407.pdf) *code*
8. **Graph Convolutional Networks for Temporal Action Localization** `ICCV2019` [*paper*](https://arxiv.org/pdf/1909.03252) [*code*](https://github.com/Alvin-Zeng/PGCN)
9. **Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?** `CVPR2018` [*paper*](https://arxiv.org/pdf/1711.09577.pdf) [*code*](https://github.com/kenshohara/3D-ResNets-PyTorch)
10. **A Closer Look at Spatiotemporal Convolutions for Action Recognition** `CVPR2018` [*paper*](https://arxiv.org/pdf/1711.11248.pdf) [*code*](https://github.com/facebookresearch/VMZ)
11. **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition** `AAAI2018` [*paper*](https://arxiv.org/pdf/1801.07455.pdf) [*code*](https://github.com/yysijie/st-gcn)
12. **Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation** `IJCAI2018` [*paper*](https://arxiv.org/pdf/1804.06055) [*code*](https://github.com/huguyuehuhu/HCN-pytorch)
13. **Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset** `CVPR2017` [*paper*](https://arxiv.org/pdf/1705.07750.pdf) [*code*](https://github.com/deepmind/kinetics-i3d)
14. **Action Recognition using Visual Attention** `ICLR2016` [*paper*](https://arxiv.org/pdf/1511.04119v2.pdf) [*code*](https://github.com/kracwarlock/action-recognition-visual-attention)
15. **Action Recognition with Trajectory-Pooled Deep-Convolutional Descriptors** `CVPR2015` [*paper*](https://arxiv.org/pdf/1505.04868) [*code*](https://github.com/wanglimin/TDD)
16. **Two-Stream Convolutional Networks for Action Recognition in Videos** `NIPS2014` [*paper*](https://arxiv.org/pdf/1406.2199.pdf) [*code*](https://github.com/jeffreyyihuang/two-stream-action-recognition)



### Speech Recognition

1. **State-of-the-art Speech Recognition With Sequence-to-Sequence Models** ` ICASSP2018` [*paper*](https://arxiv.org/pdf/1712.01769.pdf) *code*
2. **Lip Reading Sentences in the Wild** `CVPR2017` [*paper*](https://arxiv.org/pdf/1611.05358.pdf) [*code*](https://github.com/ajinkyaT/Lip_Reading_in_the_Wild_AVSR)
3. **Listen, Attend and Spell** `ICASSP2016` [*paper*](https://arxiv.org/pdf/1508.01211v2.pdf) [*code*](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)
4. **Deep speech 2: End-to-end speech recognition in english and mandarin** `ICML2016` [*paper*](http://proceedings.mlr.press/v48/amodei16.pdf) [*code*](https://github.com/PaddlePaddle/DeepSpeech)
5. **Attention-Based Models for Speech Recognition** `NIPS2015` [*paper*](https://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf) *code*
6. **Convolutional Neural Networks for Speech Recognition** `TASLP2014` [*paper*](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/CNN_ASLPTrans2-14.pdf) [*code*](https://github.com/cmaroti/speech_recognition)
7. **Hybrid speech recognition with Deep Bidirectional LSTM** `ASRU2013` [*paper*](https://www.cs.toronto.edu/~graves/asru_2013.pdf) *code*
8. **New types of deep neural network learning for speech recognition and related applications: an overview** `ICASSP2013` [*paper*](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6639344) *code*
9. **Speech Recognition with Deep Recurrent Neural Networks** `ICASSP2013` [*paper*](https://arxiv.org/pdf/1303.5778.pdf) [*code*](https://github.com/lucko515/speech-recognition-neural-network)



### Video Captioning

1. **Video Description A Survey of Methods, Datasets and Evaluation Metrics** `ACM Computing Surveys2019` [*paper*](https://arxiv.org/pdf/1806.00186.pdf) *code*
2. **Spatio-Temporal Dynamics and Semantic Attribute Enriched Visual Encoding for Video Captioning** `CVPR2019` [*paper*](https://arxiv.org/pdf/1902.10322.pdf) *code*
3. **Reconstruction Network for Video Captioning** `CVPR2018` [*paper*](https://arxiv.org/pdf/1803.11438.pdf) [*code*](https://github.com/hobincar/RecNet)
4. **Multimodal Memory Modelling for Video Captioning** `CVPR2018` [*paper*](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_M3_Multimodal_Memory_CVPR_2018_paper.pdf) *code*
5. **Interpretable Video Captioning via Trajectory Structured Localization** `CVPR2018` [*paper*](https://zpascal.net/cvpr2018/Wu_Interpretable_Video_Captioning_CVPR_2018_paper.pdf) *code*
6. **Video Captioning with Transferred Semantic Attributes** `CVPR2017` [*paper*](https://arxiv.org/pdf/1611.07675.pdf) *code*
7. **Video Paragraph Captioning Using Hierarchical Recurrent Neural Networks** `CVPR2016` [*paper*](https://arxiv.org/pdf/1510.07712.pdf) *code*
8. **Jointly Modeling Embedding and Translation to Bridge Video and Language** `CVPR2016` [*paper*](http://openaccess.thecvf.com/content_cvpr_2016/papers/Pan_Jointly_Modeling_Embedding_CVPR_2016_paper.pdf) *code*
9. **Describing Videos by Exploiting Temporal Structure** `ICCV2015` [*paper*](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Describing_Videos_by_ICCV_2015_paper.pdf) [*code*](https://github.com/yaoli/arctic-capgen-vid)