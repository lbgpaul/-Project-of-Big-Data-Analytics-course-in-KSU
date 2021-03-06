# Handling-big-NetFlow-data-and-Classification

A graduate student project in KSU.

By Inchan Hwang and Pochun Lu

Program language: python

Data base: [Challenge 2013: Mini-Challenge 3] (http://vacommunity.org/VAST)

Abstract— 

The application of Deep Neural Network into the classification of Netflow packets into an ordinary and malicious one is an interesting challenge. The automation of its classification with a neural network promises high accuracy and speed for a massive traffic monitoring where human effort cannot be effectively practiced. Compared with an existing CNN based classification, our proposed model is simpler but denotes equal or higher accuracy on the classifications of Netflow data.

we did not collect the NetFlow dataset by ourselves, but we utilized the dataset that had already been presented in VAST 2013 challenge committee, which can be download from VAST 2013 website. They provide two different datasets, Netflow data and Network health and status data. 

This dataset is the main one used to implement the analysis. This dataset contains approximately 70 million numbers of the NetFlow data, with 18 network features for each row. The network features are IP address, port number, timestamp, etc.

Accordingly to the document that provided by VAST 2013 committee, this dataset was generated by a health monitoring program. As the result, this dataset shows the status of the network flow into  four categories: 1 (Good), 2 (Warning), 3 (problem), 4 (unknown).

After combining the Network flow data with the status in the second dataset, the complete dataset with 18 features, and the status of each row has been obtained.

Model:
see the python file


Result:

With the high performance platform “Google Colaboratory” and the embedding feature data from tensorflow, the average training time per epoch is about 9 second. The final results, after training 20 epochs, we get in this experiment is 98% accuracy in both training and testing datasets (figure 2).  We can see that, for the training set, the loss and accuracy values showed high accuracy during the train. The convergence of the data happened in a really early stage, and finally can get the really high accuracy result.

<img src="images/accuracy.png" width = "500" >
<img src="images/loss.png" width = "500" >


Overfitting:

The data may have an overfitting problem. That is probably the reason that the validation loss doesn’t improve with the training dataset. The reason could be, too much features we had, need more regularization, or due to insufficient amount of data.
In this project, the model has already achieved over 95% of accuracy, so we don’t think regularization could improve the accuracy further.
About regularization techniques, in the paper[3] shows that L2 Regularization is a better method than L1 one to perform  regularization for a neural network. However, the result did not show too much difference after we add the method of L2 regularization.


Data Unbalanced:

As the result, we think the most possible reason caused this problem is lack of data.  In our dataset, the classes samples are really unbalanced: for class “1 (good)” there are 292,901 out of 300,000 number of dataset, which is the dominance class. The “2 (warning)” class only have 6,164 number of data. The “3 and 4” classed only had 971 instance. Due to the unbalance of the dataset, when splitting the data, it might give too much same classed instance to the validation dataset, even we have already called the shuffling function. KFold cross validation technique will be applied to the model in the future to clear out this issue.


DISCUSSION:

Overall, we conducted a partial experiment to analysis the Netflow. However, in some of our process there can be some discussions.

1.	The process of checking data status

About the processing to check the status of each data, we just compared the IP address, but in the real world, there might be more information to consider the Netflow data’s maliciousness.

2.	Data pre-processing method

In this paper, the main pre-processing is embedding text features. We have used the tensorflow feature column method. However, we have limited knowledge about this process, which make our model inflexible. It has given us some difficult time to build the model.

CONCLUSION:

For the over the entire experiment, we think we have gained more knowledge about handling the Netflow data, data pre-processing, and building the Deep neural network. We obtained high accuracy from the model we built, which is about 97%. Because of the limitation of the knowledge in neural network, Netflow data, and statistics, we might have not chosen the best function and the methods. These are the direction for our future research.

