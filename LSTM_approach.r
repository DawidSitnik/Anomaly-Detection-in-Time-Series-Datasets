file_path = '/home/andy/Downloads/dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/'
df_full_with_anomalies <- data.frame()
df_full <- data.frame()
df_no_anomalies <- data.frame()

#reading csv files
for(i in 1:2){
  #reading csv
  name = paste(c(file_path, "real_", i, ".csv"), collapse = "")
  df = read.csv(name, encoding = "ISO-8859-1", header=TRUE)
  df_full_with_anomalies = rbind(df_full_with_anomalies, df)
  
  #delete anomalies from dataset
  df_no_anomalies<-df[(df$is_anomaly == 0),]
  df_no_anomalies<- rev(df_no_anomalies)
  keeps = c("value")
  df_no_anomalies = df_no_anomalies[keeps]
  
  df_full = rbind(df_full, df_no_anomalies)
  rownames(df_full) <- NULL
  plot(df_full$value, cex = 0.0001, type='o')
}

timesteps <- 5
n_features <- 1

#creates attributes like value for t-1, ..., t-lookback for each value in the row
temporalize <- function(X, lookback){
  output_X <-list()

  for(i in 0:(dim(X)[1]-lookback-1)){
    t <- list()
    for (j in 1:(lookback)){
      #Gather past record within the range of lookback period
      t[[length(t)+1]] <- X$value[i+j+1]
    }
    output_X[[length(output_X)+1]] <-t
  }
  return(output_X)
}

X <- temporalize(df_full, timesteps)
X <- data.frame(matrix(unlist(X), nrow=length(X), byrow=T))

x_train <- X[0:1000, ]
print(dim(x_train))
x_train_3D <- array(x_train, c(dim(x_train)[1], timesteps, 1))

y_train = df_full[1000:1400,]

#library(keras)

#Due to problems with inputting 3d matrix into LSTM as input,
#part of the model responsible for teaching neural networs was executed in Python

#model <- keras_model_sequential() %>% 
#  layer_lstm(units = 128,  activation='relu', return_sequences=TRUE) %>% 
#  layer_lstm(units = 64, activation='relu', return_sequences=TRUE) %>% 
#  layer_lstm(units = 64, activation='relu', return_sequences=TRUE) %>% 
#  layer_lstm(units = 128, activation='relu', return_sequences=TRUE) %>% 
#  time_distributed(layer_dense(units=n_features)) %>% 
#  compile(optimizer='adam', loss='mse' ) %>% 
#  fit(x_train_3D, x_train_3D , epochs=5, batch_size=5, verbose=0)

#reading file with result created in python
df_final = read.csv("/home/andy/Downloads/dataset/ydata-labeled-time-series-anomalies-v1_0/predicted_values_from_python.csv", encoding = "ISO-8859-1", header=TRUE)

df_anomalies <- df_final[(df_final$is_anomaly == 1),]
df_no_anomalies <- df_final[(df_final$is_anomaly == 0),]


library(dplyr)
df_anomalies$r_quantile <- ntile(df_anomalies$abs_error, 100) 
df_no_anomalies$r_quantile <- ntile(df_no_anomalies$abs_error, 100) 

plot(aggregate(df_anomalies$abs_error, by=list(df_anomalies$r_quantile), FUN=mean), type='l', col='red', main='Quantiled Abs Error for Anomalies and No Anomalies', xlab = 'Quantile', ylab='Abs Error')
lines(aggregate(df_no_anomalies$abs_error, by=list(df_no_anomalies$r_quantile), FUN=mean), col='green')
