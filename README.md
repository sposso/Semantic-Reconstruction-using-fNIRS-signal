# Semantic-Reconstruction-using-fNIRS-signal
Semantic reconstruction of language from non-invasive brain recordings aims to decode the meaning of the words or sentences from neural activity patterns recorded using non-invasive neuroimaging techniques such as functional near-infrared spectroscopy (fNIRS). Previous studies have demonstrated that fNIRS contains information for language decoding. Unlike the classification approach that restricts the decoding capability of the models to a predefined set, we train our  model (referred to as "decoder")  to reconstruct the original vector representation of certain perceived concepts from the neural activity. In this study, we use Bidirectional Long Short-Term Memory (BiLSTM)-based decoder that learns to map from neural activation back to the stimuli domain. The decoding performance of the BiLSTM-based decoder is measured by computing the matching score between the generated output and the ground truth.

## fNIRS Data description: 

The fNIRS data used in this project was released in the paper [Brain decoding using fNIRS](https://ojs.aaai.org/index.php/AAAI/article/view/17493) and can be downloaded from [here](https://github.com/caolusg/decoding_fnirs). The data includes fNIRS signals recorded while healthy subjects passively viewed audiovisual stimuli that featured a photograph of an object and a simultaneous auditory presentation of the object's name. This dataset includes signals from two experiments. The first is a pilot study comprising only eight subjects and eight audiovisual stimuli. The second experiment is a large-scale decoding experiment with 7 subjects and 50  audiovisual stimuli. 

### Small dataset: Pilot study 

In this experiment, 8 participants viewed 8 objects from two semantic categories. Every object is randomly presented 12 times for three seconds with accompanying audio, followed by a 10-second rest period. The fNIRS system utilized has  **46 channels** and a sampling rate of **3.9 Hz**. The channels are arranged across three different brain regions: occipital lobe, left temporal lobe, and right temporal lobe. 


### Larger dataset: Full-scale study

In this experiment, a total of 7 healthy subjects participated. The audiovisual stimuli were extended to 50 concepts from 10 semantic categories. This experiment is divided into two sections, with 25 objects presented in each section to avoid participant fatigue. Each stimulus is presented randomly 7 times during 3 seconds, followed by a 10-second rest period after each presentation. The fNIRS system utilized **22 channels** with a sampling rate of **7.8 Hz**. The focus is mainly on the brain's left hemisphere, which is known to have two cortical areas responsible for language processing: Broca's area and Wernicke's area.

### Observation about the data :
1. The fNIRS data only contains oxygenated hemoglobin (HbO) concentrations.
2. The data provided is already preprocessed and involves the following steps: discontinuities and spike artifices are removed using nirsLab. Then, the signal is filtered by a band-pass between 0.01 and 0.1 Hz to remove physiological noise. Subsequently, the wavelength data is converted to oxygenated and deoxygenated hemoglobin concentration using the modified Beer-Lambert Law
3. The dataset does not contain the original stimuli used in the experiments.


## A bit of explanation of the repository's scripts


1. **Deep_models.py** It contains the BiLSTM-based decoder used in the experiments. 
2. **Utils.py** It contains several functions used in the paper. The most useful functions are:
  - **Regression_experiment_data**. This function loads the fNIRS  data from the large-scale study and the word embedding vectors of the stimuli for a given subject.
  + **Pilot_regression_experiment_data**. This function loads the fNIRS data from the pilot study and the word embedding vectors of the stimuli for a given subject.
  * **Train_val** This function is used to train the BiLSTM-based decoder. It includes Earlystopping and Scheduler function. 
   











