# Installation guide

## Cuda
https://developer.nvidia.com/cuda-downloads
```
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb

sudo apt-get update
sudo apt-get install cuda
sudo apt-get install nvidia-cuda-toolkit
```


### CUDNN
Download from: https://developer.nvidia.com/rdp/cudnn-download

and then, just copy the files:
cd folder/extracted/contents
sudo cp -P include/cudnn.h /usr/include
sudo cp -P lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*


## Tensorflow:

### 1. by virtual 
(recomended option by tf's developers)
```
$ sudo apt-get install python-pip python-dev python-virtualenv 
```
 Create a virtualenv environment by issuing the following command:

```
$ virtualenv --system-site-packages ~/tensorflow 
```

Activate the virtualenv environment by issuing one of the following commands:
```
$ source ~/tensorflow/bin/activate
```

The preceding source command should change your prompt to the following:

```
(tensorflow)$ 
```
To exit type:

```
(tensorflow)$ deactivate
```

Issue one of the following commands to install TensorFlow in the active virtualenv environment:

```
(tensorflow)$ sudo pip  install --upgrade tensorflow     # for Python 2.7
(tensorflow)$ sudo pip3 install --upgrade tensorflow     # for Python 3.n
(tensorflow)$ sudo pip  install --upgrade tensorflow-gpu # for Python 2.7 and GPU
(tensorflow)$ sudo pip3 install --upgrade tensorflow-gpu # for Python 3.n and GPU
```

Frequent error:
If you install the last version cuda 9 and cudnn 7 ....

```
ImportError: libcusolver.so.8.0: cannot open shared object file: No such file or directory
```
Make symbolic link to the correct version
```
cd /usr/local/cuda/lib64/
sudo ln -s libcusolver.so.9.0 libcusolver.so.8.0
sudo ln -s libcublas.so.9.0 libcublas.so.8.0
sudo ln -s libcufft.so.9.0 libcufft.so.8.0
sudo ln -s libcurand.so.9.0 libcurand.so.8.0
sudo ln -s libcudart.so.9.0 libcudart.so.8.0
sudo ln -s libcudnn.so.7 libcudnn.so.6
```