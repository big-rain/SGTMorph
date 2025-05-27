# self-supervised neuron morphology representation on graph transformer 🚀

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


![项目架构图](images/SGTMorph.png)

✨ This is the offical implementation of paper " self-supervised neuron morphology representation on graph transformer "



## System Requirements
### Hardware requirements
The training of GraphDINO requires a GPU. All trainings for the publication were performed on a NVIDIA Quadro RTX 4070ti single GPU.
The code was developed and tested on pytorch 2.1.

## Data



### Data preprocessing.
```
python3 dataloader/ACT.py
```


## Training
### supervised

```
python3 model/class_model.py
```

### self-supervised

```
python3 model/SGTMorph.py
```

The training code will write checkpoint files of the model weights to the checkpoint directory specified in the config file.


## Demos
For examples on how to load the data, train the model and perform inference with a pretrained model, see Jupyter notebooks in the


## Citation



### 安装
```bash
# 代码块必须使用语言标识
npm install your-package
