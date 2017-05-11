# Conv3D_CLSTM

## Prerequisites

1) Tensorflow-0.11 <br/>
2) Tensorlayer (commit ba30379f1b86f930d6e86e1c8db49cbd2d9aa314) or Tensorlayer-1.2.8 <br/> 
   The original Tensorlayer does not support the convolutional LSTM, so the class RNNLayer in tensorlayer/layer.py needs to be modified according to the tensorlayer-rnnlayer.py. <br/>
   
## Get the trained models
The trained models used in the paper can be obtained on the link: https://pan.baidu.com/s/1o8kDT9K Password: efm2. <br/>

## How to use the code
Use training_*.py to train the networks. Please replace the paths in the codes with your paths first. <br/>
Use testing_*.py to validate the networks. Please replace the paths in the codes with your paths first. <br/>

## Citation
Please cite the following paper if you feel this repository useful. <br/>
```
@article{Zhu2017MultimodalGR,
  title={Multimodal Gesture Recognition Using 3-D Convolution and Convolutional LSTM},
  author={Guangming Zhu and Liang Zhang and Peiyi Shen and Juan Song},
  journal={IEEE Access},
  year={2017},
  volume={5},
  pages={4517-4524}
}
```
## Contact
For any question, please contact
```
  gmzhu@xidian.edu.cn
```
