# NCI-DOE-Collab-Pilot1-Gene_Expression_Autoencoder

### Description:
The Feature Reduction capability (P1B1) shows how to build a sparse autoencoder that can compress a high dimensional expression profile into a low-dimensional vector.

### User Community:	
Primary: Cancer biology data modeling</br>
Secondary: Machine Learning; Bioinformatics; Computational Biology

### Usability:	
The current code makes heavy use of CANDLE APIs. It can be used by a data scientist experienced in python and the domain.

### Uniqueness:	
Autoencoder are not the only method for dimensionality reduction. Other techniques like principal component analysis, tSNE, or UMAP are popular for molecular data. For high dimensional input vectors auto encoder can be beneficial, but this needs to be investigated.

### Components:	
Untrained model: 
* Untrained neural network model is defined in [p1b1.model.json](https://modac.cancer.gov/searchTab?dme_data_id=).

Data:
* Processed training and test data in [MoDaC](https://modac.cancer.gov/searchTab?dme_data_id=).

Trained Model:
* Trained model is defined by combining the untrained model + model weights.
* Trained model weights are used in inference [p1b1.model.h5](https://modac.cancer.gov/searchTab?dme_data_id=).

### Technical Details:
Please refer to this [README](./Pilot1/P1B1/README.md)
