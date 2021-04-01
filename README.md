# NCI-DOE-Collab-Pilot1-Gene_Expression_Autoencoder

### Description:
The Gene Expression Autoencoder capability (Pilot 1 Benchmark 1, also known as P1B1) shows how to build three types of autoencoders autoencoder that can compress a high dimensional expression profile into a low-dimensional vector. The three types of autoencoders include regular autoencoder (ae), variational autoencoder (vae), and conditional variational autoencoder (cvae).

### User Community:
Researchers interested in the following topics:
* Primary: Cancer biology data modeling
* Secondary: Machine Learning; Bioinformatics; Computational Biology

### Usability:	
The current code makes heavy use of CANDLE APIs. It can be used by a data scientist experienced in python and the domain.

### Uniqueness:	
Autoencoder are not the only method for dimensionality reduction. Other techniques like principal component analysis, t-distributed Stochastic Neighbor Embedding (tSNE), and Uniform Manifold Approximation and Projection (UMAP) are popular for molecular data. For high dimensional input vectors auto encoder can be beneficial, but this needs to be investigated.

### Components:	
Untrained model: 
* Untrained autoencoders are defined in [p1b1.ae.model.json](https://modac.cancer.gov/searchTab?dme_data_id=), [p1b1.vae.model.json](https://modac.cancer.gov/searchTab?dme_data_id=), and [p1b1.cvae.model.json](https://modac.cancer.gov/searchTab?dme_data_id=).

Data:
* Processed training and test data in [MoDaC](https://modac.cancer.gov/searchTab?dme_data_id=).

Trained Model:
* Trained model is defined by combining the untrained model + model weights.
* Trained model weights [p1b1.ae.weights.h5](https://modac.cancer.gov/searchTab?dme_data_id=), [p1b1.vae.weights.h5](https://modac.cancer.gov/searchTab?dme_data_id=), and [p1b1.cvae.weights.h5](https://modac.cancer.gov/searchTab?dme_data_id=) are used in inference.

### Technical Details:
Refer to this [README](./Pilot1/P1B1/README.md).
