# Geometry Aware Convolutional Filters for Omnidirectional Images Representation

Implemtentation of the *Geometry Aware Convolutional Filters for Omnidirectional Images Representation* ICML 2019 paper

## Usage

This code implements the classification experiment described in the paper. In order to train a model download the dataset, as described in the following section and run
```
python classification/run_classification.py --exp=<exp_type>
```
* `exp_type` - type of the experiment which can be one of the following: [`cubic`|`fisheye`|`spherical`|`modspherical`]
* by default the experiment with cube-map projection will be executed
* you may need to adjust `classification\config` in order to run custom experiments

## Datasets

Download the datasets as follows:
* Cube-map projection [`94.7 MB`] -- required for running the code with `--exp=cubic`
```
cd data
wget --no-check-certificate -O MNISTcubic.zip https://drive.switch.ch/index.php/s/sVe1wFtqaVRwmqn/download
unzip MNISTcubic.zip
rm MNISTcubic.zip
cd ../
```

* Fish-eye projection [`62.1 MB`] -- required for running the code with `--exp=fisheye`
```
cd data
wget --no-check-certificate -O fisheye.zip https://drive.switch.ch/index.php/s/WSEy61zestEVyAQ/download
unzip fisheye.zip
rm fisheye.zip
cd ../
```

* Spherical projection [`34.5 MB`] -- required for running the code with `--exp=spherical`
```
cd data
wget --no-check-certificate -O MNISTomni.zip https://drive.switch.ch/index.php/s/5Kg8DTmhMep3iXi/download
unzip MNISTomni.zip
rm MNISTomni.zip
cd ../
```

* Modified Spherical projection [`131.8 MB`] -- required for running the code with `--exp=modspherical`
```
cd data
wget --no-check-certificate -O MNISTrandom_projection.zip https://drive.switch.ch/index.php/s/vFsZY38smcu7jA6/download
unzip MNISTrandom_projection.zip
rm MNISTrandom_projection.zip
cd ../
```

## References

If you are using the code please cite the following paper:
```
@inproceedings{KhasanovaICML19,
	author    = {Reanta Khasanova and Pascal Frossard},
	title     = {Geometry Aware Convolutional Filters for Omnidirectional Images Representation},
	booktitle = {International Conference on Machine Learning},
	year      = {2019}
}
```
