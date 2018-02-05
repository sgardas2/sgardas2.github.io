setwd('C:/SravansData/mcsds/aml')
library(klaR) # including library klar
library(caret)  # including library Caret
wdat<-read.csv('pima-indians-diabetes.data.txt',, header=FALSE)
ywdat<--wdat[,9]
split=0.80
trainIndex <- createDataPartition(ywdat, p=split, list=FALSE)
data_train <- wdat[ trainIndex,]
nrow(data_train)
data_test <- wdat[-trainIndex,]
nrow(data_test)
x = data_train[,-9]
y = data_train$V9
model = train(x,as.factor(y),'nb',trControl=trainControl(method='cv',number=10))
x_test=data_test[,-9]
y_test=data_test$V9
modelpredictnb=predict(model$finalModel,x_test)
print("accuracy for NaiveBayes")
print(confusionMatrix(modelpredictnb$class,y_test))


#Confusion Matrix and Statistics
#
#          Reference
#Prediction  0  1
#         0 97 17
#         1 13 26
#                                          
#               Accuracy : 0.8039          
#                 95% CI : (0.7321, 0.8636)
#    No Information Rate : 0.719           
#    P-Value [Acc > NIR] : 0.01037  

#	With SVM

modelsvm=svmlight(x,as.factor(y),pathsvm="C:/Users/gardask/Downloads/svm_light_windows64")
svmpredict=predict(modelsvm,x_test)
print("accuracy for SVMlight")
print(confusionMatrix(svmpredict$class,y_test))


#Confusion Matrix and Statistics

#          Reference
#Prediction  0  1
#         0 99 21
#         1 11 22
#                                          
#               Accuracy : 0.7908          
#                 95% CI : (0.7178, 0.8523)
#    No Information Rate : 0.719           
#    P-Value [Acc > NIR] : 0.02692         
                                     