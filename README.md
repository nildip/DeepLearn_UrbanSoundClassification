# DeepLearn_UrbanSoundClassification
## Using 1D CNN variants to classify sound files

This dataset contains 1302 labeled sound recordings. Each recording is labeled with the start and end times of sound events from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. Each recording may contain multiple sound events, but for each file only events from a single class are labeled. The classes are drawn from the urban sound taxonomy. 

The dataset can be downloaded from here: https://urbansounddataset.weebly.com/urbansound.html

For our experiment we will use the python library - librosa - to extract features from the sound files and then use 1D CNNs to classify the sounds.
A diagramatic representation of the models used and accuracy achieved by each model is given below:

### 1D CNN using extracted features as single concatenated list




### 1D CNN using extracted features as seperate lists
