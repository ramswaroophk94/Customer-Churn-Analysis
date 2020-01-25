
#For Random Forest and Decision tree model plots

library(RWeka)
library(party)
library(partykit)
library(sampling)
library(rpart)
library(rpart.plot)
library(pROC)
library(e1071)
library(corrplot)
library(class)
library(fpc)
library(kknn)
library(adabag)
library(gmodels)
library(boot)
library(stats)
library(ggplot2)
library(factoextra)
library(dplyr)
library(plotly)
library(dplyr)
library(httr)
library(MASS)
library(ggplot2)
library(caret)
library(MASS)
library(randomForest)
library(plyr)
tele<-read.csv('C:/Users/Jay/Desktop/WA_Fn-UseC_-Telco-Customer-Churn.csv')
dim(tele)
str(tele)
attach(tele)
#omit na values#
sapply(tele,function(x) sum(is.na(x)))
tele <- tele[complete.cases(tele),]
sapply(tele, function(x) sum(is.na(x)))
#change values to make sub categories consistent#
tele$SeniorCitizen<-as.factor(mapvalues(tele$SeniorCitizen,from=c(0,1),to=c('No','Yes')))
tele$MultipleLines<-as.factor(mapvalues(tele$MultipleLines,from=c('No phone service'),to=c('No')))
tele$OnlineSecurity<-as.factor(mapvalues(tele$OnlineSecurity,from=c('No internet service'),to=c('No')))
tele$OnlineBackup<-as.factor(mapvalues(tele$OnlineBackup,from=c('No internet service'),to=c('No')))
tele$DeviceProtection<-as.factor(mapvalues(tele$DeviceProtection,from=c('No internet service'),to=c('No')))
tele$TechSupport<-as.factor(mapvalues(tele$TechSupport,from=c('No internet service'),to=c('No')))
tele$StreamingTV<-as.factor(mapvalues(tele$StreamingTV,from=c('No internet service'),to=c('No')))
tele$StreamingMovies<-as.factor(mapvalues(tele$StreamingMovies,from=c('No internet service'),to=c('No')))
min(tele$tenure)
max(tele$tenure)
#since tenure are from 1 month to 72 month, so we can split it into 6 different class of 1 yr to 6 yrs#
tenure1<-function(tenure){
  if (tenure>=0&tenure<=12){
    return('1 year')
  }else if (tenure>12&tenure<=24){
    return('2 years')
  }else if (tenure>24&tenure<=36){
    return('3 years')
  }else if (tenure>36&tenure<=48){
    return('4 years')
  }else if (tenure>48&tenure<=60){
    return('5 years')
  }else if (tenure>60&tenure<=72){
    return('6 years')
  }
}
tele$grouptenure<-sapply(tele$tenure,tenure1)
tele$grouptenure<-as.factor(tele$grouptenure)
#remove the columns that is useless in analysis#
tele$customerID=NULL
tele$tenure=NULL
#make corr plot to see which are correlated#
numericfactor<-sapply(tele,is.numeric)
corrplot(cor(tele[,numericfactor]))
#remove 1 corr factor#
tele$MonthlyCharges<-NULL
str(tele)

# split the data into traing set and testing set#
set.sead(2018)
training<-tele[createDataPartition(tele$Churn,p=0.7,list=FALSE),]
testing<-tele[-createDataPartition(tele$Churn,p=0.7,list=FALSE),]
training
#random forest#
rf1<-randomForest(Churn~.,data=training)
print(rf1)
#prediction by random forest#
pred_rf1<-predict(rf1,testing)
table(Predicted=pred_rf1,Actual=testing$Churn)
#tuning the rf model#
tunerf1<-tuneRF(training[, -18], training[, 18], stepFactor=0.5, plot=TRUE,ntreeTry=200, trace=TRUE, improve=0.05)
#so choose mtry=2 because obb error lower#
rf2<-randomForest(Churn ~.,data=training,mtry = 2,importance=TRUE,proximity=TRUE)
#predict after tuning#
pred_rf2<-predict(rf2, testing)
table(Predicted=pred_rf2,Actual=testing$Churn)
#error rate =20.43%#
#feature importance#
varImpPlot(rf2, sort=T, n.var=10, main = 'Top Feature Importance')
#So choose top 5 element to analyse in the decision tree method#
#Decision Tree Modeling#
tree <- ctree(Churn ~ Contract+grouptenure+PaperlessBilling+PaymentMethod+InternetService, training)
plot(tree)
#Decision tree prediction and accuracy#
pred_tree1<-predict(tree,testing)
table(Predicted=pred_tree1,Actual=testing$Churn)
table_tree<-table(Predicted=pred_tree1,Actual=testing$Churn)
accuracy<-sum(diag(table_tree))/sum(table_tree)
accuracy
#accuracy=0.7998#