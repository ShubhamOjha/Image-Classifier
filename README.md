# Image-Classifier : Udacity Data Science Project
In this project, we'll train an image classifier to recognize different species of flowers. We can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content


The project includes two files `train.py` and `predict.py`. The first file, `train.py`, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image.

* Train a new network on a data set with `train.py`

  * Basic usage: `python train.py data_directory`
  * Prints out training loss, validation loss, and validation accuracy as the network trains
  * Options:
    * Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
    * Choose architecture: `python train.py data_dir --arch "vgg13"`
    * Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
    * Use GPU for training: `python train.py data_dir --gpu`
    
* Predict flower name from an image with `predict.py` along with the probability of that name. That is, we'll pass in a single image /path/to/image and return the flower name and class probability.

  * Basic usage: `python predict.py /path/to/image checkpoint`
  * Options:
    * Return top KK most likely classes: `python predict.py input checkpoint --top_k 3`
    * Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
    * Use GPU for inference: `python predict.py input checkpoint --gpu`
