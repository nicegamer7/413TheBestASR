# Automatic Speech Recognition with CRNN - CSC413 Project

This project explores the use of CRNN model to generate transcripts of English speeches from the Common Voice Corpus 19.0 dataset.

## Table of Contents
- [**Requirements**](#requirements)
- [**Audio preprocessing**](#audio)
- [**Data preparation**](#data)
- [**Model Architecture**](#model)
- [**Accuracy and Decoder**](#accuracy)
- [**Training**](#train)

## Requirements
The dependencies needed for this project are:
- Python >= 3.7
- pandas
- numpy
- torch, torchaudio, torchtext
- librosa
- jiwer
  
```bash
pip install matplotlib librosa numpy pandas torch==1.11.0 torchvision==0.12.0 torchtext==0.12.0 jiwer
```
## Audio preprocessing
## Data preparation
A subset of data from the original dataset are converted into a pandas dataframe. Then, the spectrograms are mapped with their corresponding sentences by matching the file names.
```python
# Dataframe of dataset
df = pd.read_csv("final_data.csv")
# Match input images with target sentences
for image in os.listdir(folder_path):
  sentence = df[df['path']==image[:-4] + '.mp3']['sentence'].item()
  data.append((folder_path + "/" + image, sentence))
```
Finally, we have a dataset of 10000 data points, with spectrograms as inputs and sentences as labels. The images were transformed and resized to 128x128 tensors, suitable for CNN training.

On the other han, since we are working with RNN, the sentences were converted to indices by first constructing a vocabulary object, then tokenizing the target sentences. For this step and the collate batch step, we referred to Lab 10's code.

## Model Architecture
We first created an ASR class, which represents the object model that will be trained and evaluated in this project. Appropriate layer objects such as nn.Conv2d, nn.RNN, nn.ReLu or nn.MaxPool were incorporated such that the ASR class accurately portrays our model architecture. 

## Accuracy and Decoder
A Greedy Decoder was implemented to decode the output indices to readable text. This function was also used to compute the accuracy between the decoded output and our targeted sentences. We followed the implementation of the GreedyDecoder function from this source: https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/.

Lastly, the compute word accuacy function compute_Wacc utilized the wer function from jiwer library to generate a word error rate value, helping us with the quantitative evaluation of our model.
```python
def compute_Wacc(output, labels, label_lengths):
    # Decode the predictions
    decodes, targets = GreedyDecoder(output, labels, label_lengths)
    error = wer(targets, decodes)
    return 1 - error
```

## Training
For training, we followed the standard format of a train_model function from the labs. However, we implemented the CTC loss with torch.nn.CTCLoss. This function requires log_probs (log probabilities of outputs across the vocabulary), labels (list of labels), input_lengths (the lengths of the output sequences in batch), and label_lengths (the lengths of label sequences in batch).
In particular, input_lengths and label_lengths are arrays of length batch_size, in which the items are the sequence lengths. 
```python
z = model(images).transpose(0, 1)
log_prob = z.log_softmax(2)
input_lengths = torch.full(size=(z.shape[1],), fill_value=z.shape[0], dtype=torch.int32).tolist()
loss = ctcloss(log_prob, labels, input_lengths, label_lengths)
loss.backward()
```

During the training process, the training and loss accuraces were recorded and plotted with matplotlib for better visualization.

