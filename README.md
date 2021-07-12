![logo](logo.jpg)
# CRISPR-VAE

[CRISPR-VAE](https://www.biorxiv.org/content/early/2021/07/06/2021.07.05.451176) is a framework used for interpreting the explanation of gRNAs efficiency predictors, using an efficiency-aware sequence generator that allows low-level editing control.
This repo contains the codes that are used to implement, train, and use CRISPR-VAE in Keras, including the synthetic data used in the paper.

The codes can be easily operated with one-line command as shown below.

## Dependencies
* Python 3.7.10
* Tensorflow 2.1.0
* OpenCV 3.4.2
* Openpyxl 
* scikit-learn 0.24.2
* matplotlib
* numpy

To ensure compatability, install all packages at once. Example:
```
conda install opencv scikit-learn tensorflow openpyxl matplotlib
```
Note: Installing opencv takes care of python and numpy.

## Usage
1. The main file is crispr_vae.py, which can be operated as the following:

```
python crispr_vae.py --trained_mdl --ready_synth --heatmaps --MSM --CAM
```
All options are boolean, but must be inputted as strings; '0' for False, and '1' for True.

* ``` --trained_mdl ```
  - Choose whether to load the weights of CRISPR-VAE ('1'), or train it from scratch ('0'). The default is '1'.

* ``` --ready_synth ```
  - Choose whether to use existing synthetic data ('1'), or generate new ones ('0'). The default is '1'.

* ``` --heatmaps ```
  - Choose whether to generate latent space structure heatmaps for confirmation purposes. The default is no ('0').

* ``` --MSM ```
  - Choose whether to generate Mer Significance Maps (MSMs). The default is yes ('1').

* ``` --CAM ```
  - Choose whether to generate Class Activation Maps (CAMs). The default is yes ('1').

## Example Run
```
python crispr_vae.py --CAM '0'
```
This will run the code according to the defualt options, but will exclude generating the CAMs.

## Output directory
All available outputs are located in /Files/outputs

## Citation
If you find this repo useful, please include the following citation in your work. 

> @article {Obeid2021.07.05.451176,
	author = {Obeid, Ahmad and Al-Marzouqi, Hasan},
	title = {CRISPR-VAE: A Method for Explaining CRISPR/Cas12a Predictions, and an Efficiency-aware gRNA Sequence Generator},
	elocation-id = {2021.07.05.451176},
	year = {2021},
	doi = {10.1101/2021.07.05.451176},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/07/06/2021.07.05.451176},	
	journal = {bioRxiv}
}

The current version of the paper can be found in bioRxiv.



  


