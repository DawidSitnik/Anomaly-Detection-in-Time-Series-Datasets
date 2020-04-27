# Anomaly Detection in Time Series Datasets
*authors: Dawid Sitnik, Władysław Sinkiewicz*

**File Description**

* LSTM_approach.ipynb - jupyter notebook in which the whole solution is made
* LSTM_approach.r - the same solution made in R with extraction of the fragment responsible for learning LSTM network.

## The Aim of The Project
The main goal is to detect anomalies in time series dataset. As the dataset we decied to choose data shared by Yahoo called *'A Benchmark Dataset for Time Series Anomaly Detection'*, which is the real traffic data from Yahoo servers. 

The data can be accessed from this url:
[Yahoo dataset](https://yahooresearch.tumblr.com/post/114590420346/a-benchmark-dataset-for-time-series-anomaly?fbclid=IwAR31SaUo48kFzUCeYPFDfVGRKyqYPW3vmY0XDuci7uIYM-XrrW86QXGerrY)

## Our Approach
To detect anomalies we are going to create some models whihch will be learned on the dataset which doesn't consist any anomalies. Than we will make predictions on the dataset which consists also the data with anomaly. Assuming that our models will work properly, predictions which values are much different than real values will be treated as anomalies.

In our project we would like to compare classical approaches of modeling with more modern one. The first group of sollutions will be based on transforming dataset into its vector representation of time series (for example - values history for certain period with its eventual aggregation in smaller sub-windows in different variants) and then use one of classical alghoritms like linear regression, random forest classifier etc. to create the model. In the second approach we will use LSTM neural network which will work only on historical values from the time series.

## Dataset
This dataset is provided as part of the Yahoo! Webscope program, to be used for approved non-commercial research purposes by recipients who have signed a Data Sharing Agreement with Yahoo! Dataset contains real and synthetic time-series with labeled anomalies. Timestamps are replaced by integers with the increment of 1, where each data-point represents 1 hour worth of data. The anomalies are marked by humans and therefore may not be consistent.

**The dataset fields are:**
* *timestamp*
* *value*
* *is_anomaly*
    
The is_anomaly field is a boolean indicating if the current value at a given timestamp is considered an anomaly.

**Dataset snippet:**

*1,83,0*

*2,605,0*

*3,181,0*

*4,37,0*

*5,45,1*

The dataset before preprocessing:
<p align="center">
  <img src = "https://imgur.com/qkjA3oo.png"/>
</p>

### Data Preprocessing
Before feeding the network with data we should preprocess our data. The first thing which comes to the mind is to detect trends and seasonality and delete it from the signal. However, in this case there was no seasonality nor trend. Than we checked if the data doesn't contain any null falues and outlayers, but fortunatelly it didn't.

Another thing which can be done is data normalization. In our purpose we decided to normalize our data due to this formule:
```
df = (df - df.mean()) / (df.max() - df.min())
```

As we can see, our signal is quite complex - it changes its values drastically, so we also have tried to smooth it a little bit using rooling window technique trying different values of window. The window size which gave us the best result was 5.

The preprocessed signal can be seen at the picture below:

<p align="center">
  <img src = "https://imgur.com/oSG2zG7.png"/>
</p> 

So finally description of our data is:
<p align="center">
  <img src = "https://imgur.com/BMTJWsV.png"/>
</p>




### LSTM Neural Network Approach
A powerful type of neural network designed to handle sequence dependence is called recurrent neural networks. The Long Short-Term Memory network or LSTM network is a type of recurrent neural network used in deep learning because very large architectures can be successfully trained.

Before feeding the network with the data we needed to extend our dataset by assigning new attributes to each value. The new attributes are previous values of time series, so for value recorded at time t we extended it of values from t-1, t-2 ... t-n. In this case we got dataframe of size (initial length of dataframe x n). We couldn't create all of the attributes for last n observations from each subdataset so we decided to not include them in our training dataset. 

Because of the LSTM neural network nature, we had to reshape our data one more time to get its final dimension equal to (initial length of dataframe x n x 1). It had to be done, because those types of networks opperates only on 3D vectors. The fraction of the training dataset can be seen at the picture.

<p align="center">
  <img src = "https://imgur.com/kT1A5pH.png"/>
</p>

The final architecture of the LSTM NN looks like:
<p align="center">
  <img src = "https://imgur.com/oK2PkrI.png"/>
</p>

Its graphical ilustration can be seen below:
<p align="center">
  <img src = "https://imgur.com/339bam0.png"/>
</p>

We repeated learning process many times, trying different sizes of:
* n - attributes which were value from previous timestep
* window size for smoothing the signal
* epochs 

The best obtained result was for:
* n = 5
* window size = 1, so there was no need for simplyfing the signal
* 100 epochs

The learning process of network can be seen here:
<p align="center">
  <img src = "https://imgur.com/59ARm9m.png"/>
</p>

Signal prediction on training dataset:
<p align="center">
  <img src = "https://imgur.com/WbhzOD0.png"/>
</p>


Signal prediction on testing dataset:
<p align="center">
  <img src = "https://imgur.com/Dx0f3ja.png"/>
</p>

The model isn't overtrained, as it can be seen at the pictures, signal predicted basing on test dataset is as good as the one predicted on training dataset.

Than, to point the anomalies we tried to find the treshold which is the the absolute value of difference between real and predicted values. To do that we splited our results into two groups - anomaly data and normal data. The summarize of absolute errors for each group can be seen at the picture:

group without anomalies
<p align="center">
  <img src = "https://imgur.com/4HMz185.png"/>
</p>

group with anomalies
<p align="center">
  <img src = "https://imgur.com/rwVLJJg.png"/>
</p>

The max error for group without anomalies equals to 0.151, when min error for group with anomalies is 0.1639. It means ,that we can easily split the difference between those two groups if we choose a loterated error treshold sensibly.

The treshold value set to differenciate groups correctly is *12 * (standard deviation of the whole dataset)* what equals to 1.559.



