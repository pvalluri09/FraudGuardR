#Credit Card Fraud Detection on sample-Dataset
library(ranger)      #for building random forest models
library(data.table)  #for efficient data manipulation
setDTthreads(8)      #set number of threads for parallel processing
library(ggplot2)     #for creating visualizations
library(lattice)     #for additional plotting functions
library(caret)       #for machine learning model evaluation

#Read the credit card dataset from a CSV file
data<-read.csv("C:\\Users\\Vaishu\\Documents\\delete\\creditcard.csv")

#explore the dataset
data.table(data)

#Display summary satistics of the dataset
summary(data)

#Check the distribution of the class variable
table(data$class) #frequency table - count of unique values

#get column names of the dataset
names(data)

#summary satistics of the amount column
summary(data$Amount)
sd(data$Amount)
IQR(data$Amount)    #range in which 50% of the data values lie(data sorted in ascending order iqr=q3-q1)
var(data$Amount)

#scale the amouc(nt column
data$Amount<-scale(data$Amount)    #standardization transforms a data so that mean of 0 and standard deviation of 1

#Remove the first column (assumed to be the transaction ID) from the dataset
data2<-data[,-c(1)]   #creating a new dataframe excluding first column
head(data2)

#set.seed for reproductivity and split the data into training and testing sets
set.seed(12)    #generates same set of random numbers so it uses same random samples each time
library(caTools) #for data splitting 
sample_data<-sample.split(data2$Class, SplitRatio = 0.80)  #randomly splits the dataset in given proportion

train_data <- subset(data2, sample_data=TRUE)
test_data <- subset(data2, sample_data=FALSE)

#check the dimension of training and testing sets
dim(train_data)
dim(test_data)

#fit a logistic model on training data
Logistic_Model <- glm(Class~., test_data,family = binomial())   #responce~predictors class=responce and .=other variables in train.data
summary(Logistic_Model) 
plot(Logistic_Model)




Logistic_Model1 <- glm(Class~.,train_data,family = binomial())
summary(Logistic_Model1)
plot(Logistic_Model1)



#we need ROC curve visit bigquery tutorial to learn abouy ROC
library(pROC)   #for roc curve analysis

# Assuming lr.predict is the predicted probabilities for the positive class (e.g., Class == 1)
# and test_data$Class contains the true class labels (0 or 1)
lr.predict <- predict(Logistic_Model1, test_data, probability = TRUE)  #predict() - probabilities of likelihood function belonging to positive class if true=returns probabilities if false=returns predicted values
#create roc curve using true class labels and predicted probabilities
roc_curve <- roc(test_data$Class, lr.predict)  #roc-graphical rep shows the trade-off between (true positive value) and specificity(true negative value)


# Plot the ROC curve
plot(roc_curve, col = "green", main = "ROC Curve", sub = "Logistic Regression", col.main = "green", col.sub = "green")

#so we have atmost 90% accuracy but it is not the right method of doing projects like this

#fit a decision tree
library(rpart)       #for building decision tree models
library(rpart.plot)  #for plotting decision tree

desicion_model <- rpart(Class ~ ., data, method = "class") #contain decision tree(split the data based on predictor values such a way that it optimises difference between classes) model using all predicts to predict the categorial outcome rep by class variable
predicted_val <- predict(desicion_model, data,type="class") #contains predicted class labels for each observation in data dataset according to decision tree model
probability <- predict(desicion_model,data, type= 'prob') # Each row of the probability matrix corresponds to an observation in the data dataset, and each column represents the probability of belonging to a specific class.
rpart.plot(desicion_model)

#do small NN
library(neuralnet)     #for building network model
NN_model <- neuralnet::neuralnet(Class~., train_data,linear.output = FALSE)
plot(NN_model) #used to visualize various types of objects

#weights - strength of the influence once neuron on the other
#biases - allows the neuron to activate when the inputs are not significant 

predNN <- compute(NN_model,test_data) #It takes the neural network model and the dataset (or data matrix) as input and produces predictions based on the learned weights and biases in the model.
resultNN <- predNN$net.result
resultNN <- ifelse(resultNN > 0.6, 1, 0)