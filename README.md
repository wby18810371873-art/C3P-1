This is the code release accompanying the paper $C^3P$:Confusion-Guided Conformal Predictionwith Class Clustering for Compact Sets

Citation: 
```
@article{wei2026c3p,
  title={$C^3P$:Confusion-Guided Conformal Predictionwith Class Clustering for Compact Sets},
  author={Boyao Wei, Zihao Wu, Yinglong Wang},
  journal={},
  year={2026}
}
```

## Setup 

First, create a virtual environment and install the necessary packages by running

```
conda create --name env
conda activate env
pip install -r requirements.txt
```

To make the environment accessible from Jupyter notebooks, run

```
ipython3 kernel install --user --name=conformal_env
```

This adds a kernel called `conformal_env` to your list of Jupyter kernels.

Download the datasets by running

```
sh download_data.sh
```

which will create a folder called `data/` and download the data described in the following section. 

## Data description

1. `imagenet` (4.62 GB): `(115301, 1000)` array of softmax scores and `(115301,)` array of labels
1. `cifar-100` (0.01 GB): `(30000, 100)` array of softmax scores and `(30000,)` array of labels
1. `places365` (0.54 GB): `(183996, 365)` array of softmax scores and `(183996,)` array of labels
1. `inaturalist` (6.72 GB): `(1324900, 633)` array of softmax scores and `(1324900,)` array of labels


