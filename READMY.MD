# Image Single Class Segmentation with PyTorch Implementation of UNet
Image Single Class Segmentation is a small collection of jupyter notebooks and scripts written for the sake of learning of PyTorch and CNN. It consists of a training notebook, an inference notebook, an inference script, and a terminal script. They can be run on both GPU and CPU. It was tested on WIN 10 with RTX 3050.

# Prerequisites
Anaconda, CUDA
Env can be loaded from pytorchenv.yml

# Usage
Inference can be used for inference of a single image or b batch processing of a folder

# Data set for training
[Finding and Measuring Lungs in CT Data (https://www.kaggle.com/kmader/finding-lungs-in-ct-data)]

# To do
- [ ] Add continued training
- [ ] Add simple statistics for dataset

# Licence
[MIT](https://choosealicense.com/licenses/mit/)

# Acknowledgments
The [Swapnil Bhange(https://medium.com/@swapnilbhange98/image-classification-between-dog-and-cat-using-resnet-in-pytorch-fdd9fdfac20a)] code was used as a basis for the scripts
UNet architecture is taken from [milesial(https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py)]
DiceBCELoss function for multi class segmentation with UNet was taken from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
