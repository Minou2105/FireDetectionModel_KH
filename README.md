# FireDetectionModel_KH
Capstone project for KnowledgeHut. The task is to classify images in the classes "smoke", "fire", "non fire"

# Usage

All files beside of the kaggle_notebook.ipynb were used to train the network locally. Unfortunately this did not work out as expected as my GPU is too small.
These files are more modular and can be run on any computer if wanted.

The main work was done in the kaggle_notebook.ipynb. As the name already suggests it was done using the kaggle environment where you cannot modularize your code.
Hence all code is available in this notebook. Just run it from top to bottom. But it will only work in a kaggle environment probably.

I did not save the code for all models but rather changed it manually. The configuration of each model can be found in the results folder, together with some
additional information about loss and accuracy development over the epochs and the logs of the trainings. Also the best model stored from the Endpoint callback are saved here.

The data needs to be saved in an additional folder called 'data'. I just want to mention here that the dataset as it was downloadable from the guidebook has a lot of images in
it which are corrupted or in a file format not readable by TensorFlow. Training will crash though. If the updated dataset is needed please contact me. It is to large to store it
in github.
