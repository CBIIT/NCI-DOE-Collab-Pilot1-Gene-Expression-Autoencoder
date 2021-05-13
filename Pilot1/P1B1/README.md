### Model Description
The Gene Expression Autoencoder capability (Pilot 1 Benchmark 1, also known as P1B1) is an autoencoder that can collapse high-dimensional expression profiles into low-dimensional vectors without much loss of information. The proposed network architecture includes encoding layers, dropout layers, bottleneck layer, and decoding layers. At least three hidden layers are required, including one encoding layer, one bottleneck layer, and one decoding layer. It is useful for compressing the high dimensional expression profiles into low dimensional vectors.

### Description of the Data
* Data source: RNA-seq data from GDC 
* Input dimensions: 60,484 floats; log(1+x) transformed FPKM-UQ values
* Output dimensions: Same as input
* Latent representation dimension: 1000
* Sample size: 4,000 samples (3000 training + 1000 test)
* Notes on data balance and other issues: unlabeled data draw from a diverse set of cancer types

### Expected Outcomes
* Reconstructed expression profiles
* Output range: float; same as log transformed input

### Setup
To set up the Python environment needed to train and run this model:
1. Install [conda](https://docs.conda.io/en/latest/) package manager.
2. Clone this repository.
3. Create the environment as shown below.

```bash
   conda env create -f environment.yml -n P1B1
   conda activate P1B1
   ```

To download the processed data needed to train and test the model, and the trained model files:
1. Create an account on the Model and Data Clearinghouse [MoDaC](modac.cancer.gov). 
2. Follow the instructions in the Training section below.
3. When prompted by the training and test scripts, enter your MoDaC credentials.

### Training

To train the model from scratch, execute the script [p1b1_baseline_keras2.py](p1b1_baseline_keras2.py), as follows:

```cd Pilot1/P1B1
   python p1b1_baseline_keras2.py --cp TRUE --save_path p1b1 --model vae --latent_dim 100
   ```

The script  does the following:
* Reads the model configuration parameters from [p1b1_default_model.txt](p1b1_default_model.txt).
* Downloads the training data and splits it to training/validation sets.
* Creates and trains the Keras model.
* Saves the best-trained model based on the model performance on the validation dataset.

The first time you run the script, it downloads the training and test data files. Then it caches the files for future runs.

The baseline implementation supports three types of autoencoders controlled by the `--model` parameter: regular autoencoder (`ae`), variational autoencoder (`vae`), and conditional variational autoencoder (`cvae`).

#### Example Output of Variational Autoencoder (VAE)

```
Using TensorFlow backend.

Shape x_train: (2700, 60483)
Shape x_val:   (300, 60483)
Shape x_test:  (1000, 60483)
Range x_train: [0, 1]
Range x_val:   [0, 1]
Range x_test:  [0, 1]
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 60483)        0                                            
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 2000)         120968000   input_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 600)          1200600     dense_1[0][0]                    
__________________________________________________________________________________________________
z_mean (Dense)                  (None, 100)          60100       dense_2[0][0]                    
__________________________________________________________________________________________________
z_log_var (Dense)               (None, 100)          60100       dense_2[0][0]                    
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 100)          0           z_mean[0][0]                     
                                                                 z_log_var[0][0]                  
__________________________________________________________________________________________________
model_2 (Model)                 (None, 60483)        122289083   lambda_1[0][0]                   
==================================================================================================
Total params: 244,577,883
Trainable params: 244,577,883
Non-trainable params: 0
__________________________________________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         (None, 100)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 600)               60600     
_________________________________________________________________
dense_4 (Dense)              (None, 2000)              1202000   
_________________________________________________________________
dense_5 (Dense)              (None, 60483)             121026483 
=================================================================
Total params: 122,289,083
Trainable params: 122,289,083
Non-trainable params: 0
_________________________________________________________________
None
Train on 2700 samples, validate on 300 samples

Epoch 99/100
 - 7s - loss: nan - xent: 0.4575 - corr: 0.7870 - mse: 0.0382 - val_loss: nan - val_xent: 0.4817 - val_corr: 0.7422 - val_mse: 0.0466
Current time ....751.347
Epoch 100/100
 - 7s - loss: nan - xent: 0.4575 - corr: 0.7870 - mse: 0.0382 - val_loss: nan - val_xent: 0.4818 - val_corr: 0.7420 - val_mse: 0.0466
Current time ....758.342
```

### Preliminary Performance

The current best performance in terms of validation correlation for the three types of autoencoders are as follows:

* AE: 0.78
* VAE: 0.77
* CVAE: 0.77

### Inference

To test the trained model in inference, execute the script [p1b1_infer.py](p1b1_infer.py), as follows: 

```bash
   python p1b1_infer.py --model vae --model_name p1b1
   ```
   
This script does the following:
* Downloads the trained model from MoDaC.
* Downloads the processed test dataset from MoDaC with the corresponding labels.
* Learns a representation for the test dataset.
* Reports the performance of the model on the test dataset, including mse, r2_score, and correlation.
* Creates the 2D latent representation of encoded test dataset.

#### Example Output
```
Evaluation on test data: {'mse': 0.033340786, 'r2_score': 0.15194242634880137, 'correlation': 0.8182240011660016}

```
Here is a visual example of the 2D latent representation from VAE.
![VAE latent representation](https://github.com/CBIIT/NCI-DOE-Collab-Pilot1-Gene_Expression_Autoencoder/blob/e40dc4eb0e1ab58ef50d0e3a80a265a2ba036c96/Pilot1/P1B1/images/p1b1.vae.latent.png)
