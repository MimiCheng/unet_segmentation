# unet_segmentation
This repository is to predicting whether a CT scan is of a patient who either has or will develop lung cancer within the next 12 months or not.
To do so, I will use data from LUng Nodule Analysis 2016. The dataset can be found [here](https://luna16.grand-challenge.org/Download/).

The network is trained to segment out potentially cancerous nodules and then use the characteristics of that segmentation to make predictions about the diagnosis of the scanned patient within a 12 month time frame.


# Requirement

* numpy
* scikit-image
* scikit-learn
* keras (tensorflow backend)
* matplotlib
* pydicom
* SimpleITK
