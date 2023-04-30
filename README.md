# Drowsiness Detection ML Implementation

This is a Jupyter notebook implementation of a drowsiness detection machine learning model. The code is implemented using Python programming language and various libraries are used for image processing, visualization, and machine learning purposes.

## Libraries Used
- os: Used for directory and file operations
- cv2: Used for image processing
- tqdm.notebook: Used for progress bar during data loading
- numpy: Used for numerical computations
- matplotlib.pyplot: Used for image visualization
- random: Used for random operations
- tensorflow: Used for machine learning purposes
- sklearn.metrics: Used for confusion matrix calculation
- sklearn.model_selection: Used for train-test split of dataset
- scipy.ndimage: Used for image rotation
- keras: Used for model building

## Dataset 

- used the yawn_eye_dataset_new, which is available [Kaggle](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new)
- used the Drowsiness Detection Dataset, which is available [Kaggle](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset)
- used the Drowsiness Prediction Dataset, which is available [Kaggle](https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset)


## Model Building
The init_model() function is used to build the machine learning model. It is a sequential model with a 2D convolutional layer, max pooling layer, flatten layer, and dense layer. The output layer has 6 units, which corresponds to the number of categories in the dataset.

## Conclusion
This Jupyter notebook implementation of a drowsiness detection machine learning model is an example of how machine learning can be used for image processing and classification tasks. The model is trained on a dataset with six categories, and the accuracy of the model can be improved by adding more data to the dataset and fine-tuning the model's hyperparameters.


