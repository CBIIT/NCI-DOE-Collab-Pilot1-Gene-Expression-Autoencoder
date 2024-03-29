# NCI-DOE-Collab-Pilot1-Gene-Expression-Autoencoder

### Description:
The Gene Expression Autoencoder capability (Pilot 1 Benchmark 1, also known as P1B1) shows how to build three types of autoencoders that can compress a high-dimensional expression profile into a low-dimensional vector. The three types of autoencoders include regular autoencoder (ae), variational autoencoder (vae), and conditional variational autoencoder (cvae).

### User Community:
Researchers interested in the following topics:
* Primary: Cancer biology data modeling
* Secondary: Machine Learning; Bioinformatics; Computational Biology

### Usability:	
The current code can be used by a data scientist experienced in Python and the domain. 

### Uniqueness:	
Autoencoders are not the only method for dimensionality reduction. Other techniques like principal component analysis, t-distributed stochastic neighbor embedding (tSNE), and uniform manifold approximation and projection (UMAP) are popular for molecular data. For high-dimensional input vectors, autoencoders can be beneficial.

### Components:	
The following components are in the [Gene Expression Autoencoder](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-7648032) dataset in the Model and Data Clearinghouse (MoDaC):
* Untrained model: 
  * The untrained autoencoders are defined in the model topology files: p1b1.ae.model.json, p1b1.vae.model.json, and p1b1.cvae.model.json.
* Processed Data:
  * The training data are in the P1B1.train.csv file.
  * The test data are in the P1B1.test.csvfile.  
* Trained Model:
  * The trained model is defined by combining the untrained model and model weights.
  * The trained model weights are used in inference: p1b1.ae.weights.h5, p1b1.vae.weights.h5, and p1b1.cvae.weights.h5.

### Technical Details:
Refer to this [README](./Pilot1/P1B1/README.md).
