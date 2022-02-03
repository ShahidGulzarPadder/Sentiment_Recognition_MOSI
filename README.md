# Sentiment_Recognition_MOSI
In this repository I have built and trained a multimodal deep neural network for sentiment recognition using tf.keras/pytorch. Here I have worked with the MOSI dataset, which contains more than 2000 short videos.  Every video has a single label, which is a continuous value in range [-3, +3].  For this repo, the following tasks are defined on this dataset:      regression: using the raw values between [-3, +3]     classification with 2 classes: [-3, +3] values are converted to 2 classes: negative and positive.
## Example pipeline

There is no constraint, you can preprocess the data as you wish, use any DNN (or Transformer architecture) for training, etc. The expected solution should contain data loading and visualization, preprocessing, model definition, training and evaluation.

Here, there is an **example pipeline** with such details to help you out. (You don't have to follow it):
* Download the MOSI dataset, it is available in zip format at the following link: nipg1.inf.elte.hu:8765/MOSI.zip (if the file is unreachable, write an email)
* It contains short video clips (utterance level, they are mostly between 3-10 seconds) and the labels in a csv file.
* Preprocess the data and visualize some samples
  * Extract OpenFace features from the video clips.
  * Extract openSMILE features from the audio.
  * Separate the train-valid-test subsets (train, valid, test) defined in the labels.csv file
* Define train-valid-test dataloaders
  * standardize the inputs (you can't really augment openFace outputs, but you can augment other modalities, like RGB texture)
* Define and train a model
  * Do a model fusion, by defining a visual submodel on OpenFace inputs, an acoustic submodel on openSMILE inputs, then concatenate hidden representation at a point and add a shallow network on top of it. 
  * Train the two submodels with the shallow on top together.
  * define the loss function, optimizer
  * use early stopping and a learning rate scheduler
  * train the model using the training and validation subsets
  * plot the training/validation curve
* Evaluation
  * evaluate the model on the test set
  * print/plot classification results (Binary accuracy, F1) or regression metrics (MAE, RMSE, R^2, Corr)
 ## Use GPU
Runtime -> Change runtime type

At Hardware accelerator select GPU then save it.
## Useful shortcuts
* Run selected cell: *Ctrl + Enter*
* Insert cell below: *Ctrl + M B*
* Insert cell above: *Ctrl + M A*
* Convert to text: *Ctrl + M M*
* Split at cursor: *Ctrl + M -*
* Autocomplete: *Ctrl + Space* or *Tab*
* Move selected cells up: *Ctrl + M J*
* Move selected cells down: *Ctrl + M K*
* Delete selected cells: *Ctrl + M D*
## Tip

Downloading the dataset and preprocessing may require more storage, than Colab lend you. I suggest, that you should download the data and preprocess it locally (computer/laptop/tamagotchi). Then create a pickle file, which is preferably less than 5 Gb - This can be uploaded to your Google Drive, and import it in Colab only once. (Uploading the data is also done once this way.)
After loading the pickle file, only the training and visualization stuff is expected to be done in Colab.

If you have your own resources to train, you don't have to use Colab. However, you have to make sure, that the tasks / outputs can be checked. 
