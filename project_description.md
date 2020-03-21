# Anomaly Detection in Time Series Datasets
*authors: Dawid Sitnik, Władysław Sinkiewicz*

# The Aim of The Project
The main goal is to detect anomalies in time series dataset. As the dataset we decied to choose data shared by Yahoo called *'A Benchmark Dataset for Time Series Anomaly Detection'*, which is the real traffic data from Yahoo servers. 

The data can be accessed from this url:
[Yahoo dataset](https://yahooresearch.tumblr.com/post/114590420346/a-benchmark-dataset-for-time-series-anomaly?fbclid=IwAR31SaUo48kFzUCeYPFDfVGRKyqYPW3vmY0XDuci7uIYM-XrrW86QXGerrY)

# Our Approach
To detect anomalies we are going to create some models whihch will be learned on the dataset which doesn't consist any anomalies. Than we will make predictions on the dataset which consists also the data with anomaly. Assuming that our models will work properly, predictions which values are much different than real values will be treated as anomalies.

In our project we would like to compare classical approaches of modeling with more modern one. The first group of sollutions will be based on transforming dataset into its vector representation of time series (for example - values history for certain period with its eventual aggregation in smaller sub-windows in different variants) and then use one of classical alghoritms like linear regression, random forest classifier etc. to create the model. In the second approach we will use LSTM neural network which will work only on historical values from the time series.

# Dataset
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

The dataset consists of 10 independent .csv files which combined signal is presented at the picture below: 
<p align="center">
  <img src = "https://i.imgur.com/FGLEwGT.png"/>
</p>

## Data Preprocessing
Before feeding the network with data we should preprocess our data. The first thing which comes to the mind is to detect trends and seasonality and delete it from the signal. However, in this case there is no seasonality nor trend, as we can see at the picture. 

Another thing which can be done is data normalization. In our purpose we decided to normalize our data due to this formule:
```
df = (df - df.mean()) / (df.max() - df.min())
```


# LSTM neural network approach
Those networks are type of recurrent neural networks, which complex architecture enables 
A powerful type of neural network designed to handle sequence dependence is called recurrent neural networks. The Long Short-Term Memory network or LSTM network is a type of recurrent neural network used in deep learning because very large architectures can be successfully trained.


