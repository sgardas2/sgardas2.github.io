setwd('C:/SravansData/mcsds/aml')
wdat<-read.csv('pima-indians-diabetes.data.txt', header=FALSE)
library(klaR)
library(caret)
bigx<-wdat[,-c(9)] # all the features from colum 1 to 8 and 9th contains the label
bigy<-wdat[,9] #9th colum contains the label
nbx<-bigx
for (i in c(3, 4, 6, 8))
{vw<-bigx[, i]==0 #making the features 3,4,6,8 as NA when they are 0
 nbx[vw, i]=NA
}
trscore<-array(dim=10) #declaring an array of 10 for the scores
tescore<-array(dim=10) #declaring the array of 10 for test scores
for (wi in 1:10) # for loop for 10 iteratiors for train and test split
{wtd<-createDataPartition(y=bigy, p=.8, list=FALSE) #Splitling the data by createdatapartition of Cartet package into 80% train and 20% test
 ntrbx<-nbx[wtd, ] #nbx data frame has all features
 ntrby<-bigy[wtd] # ntrby has the labels
 trposflag<-ntrby>0 #trposflag contains true is the label is 1 else it contains false
 ptregs<-ntrbx[trposflag, ] #all the rows from ntrbx which has postive label 
 ntregs<-ntrbx[!trposflag,] #all the rows from ntrbx which has negative label 
 ntebx<-nbx[-wtd, ] # ntebx contains the test data or the remaning 20% of the data
 nteby<-bigy[-wtd] #the actual labels for the test data
 ptrmean<-sapply(ptregs, mean, na.rm=TRUE) #calcuating the mean of  all the eight features which has the postive label
 ntrmean<-sapply(ntregs, mean, na.rm=TRUE) #calculating the mean of the eight features which have negative label
 ptrsd<-sapply(ptregs, sd, na.rm=TRUE) #calculating the standard deviation of the  eight features which have the postive label
 ntrsd<-sapply(ntregs, sd, na.rm=TRUE) # calculating the standard deviation of the eight features which have the negative label
 ptroffsets<-t(t(ntrbx)-ptrmean) # subtracting the mean from the training features for postive label
 ptrscales<-t(t(ptroffsets)/ptrsd) #divdiving the offsets by standard deviation for postive label
 ptrlogs<--(1/2)*rowSums(apply(ptrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) # log likelihoods using normal distribution for postive label of diabetes
 ntroffsets<-t(t(ntrbx)-ntrmean)  # subtracting the mean from the training features for Negative label
 ntrscales<-t(t(ntroffsets)/ntrsd) #divdiving the offsets by standard deviation for Negative label
 ntrlogs<--(1/2)*rowSums(apply(ntrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) # log likelihoods using normal distribution for Negative label of diabetes
 lvwtr<-ptrlogs>ntrlogs  #rows classfied as diabetes postive by the classifier
 gotrighttr<-lvwtr==ntrby #comparing the results with the actual training label
 trscore[wi]<-sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr)) # the accuarcy of the classification for the training set
 pteoffsets<-t(t(ntebx)-ptrmean) #normalize the test data with mean of training
 ptescales<-t(t(pteoffsets)/ptrsd)  #normalize the test data with the standard deviation of training
 ptelogs<--(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) # log likehoods using normal distribution for postive labels of test data
 nteoffsets<-t(t(ntebx)-ntrmean) #normalize the test data with mean of training for negative label
 ntescales<-t(t(nteoffsets)/ntrsd) #normalize the test data with the standard deviation of training for negative label
 ntelogs<--(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) # log likehoods using normal distribution for negative labels of test data
 lvwte<-ptelogs>ntelogs #rows classfied as diabetes postive by the classifier for test data 
 gotright<-lvwte==nteby #comparing the results with the actual testing label
 tescore[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright)) # the accuarcy of the classification for the test set
}

print(paste(mean(trscore), "mean accuracy of Training data"))
print(paste(mean(tescore), "mean accuracy of Testting data"))

#> mean(trscore)
#[1] 0.735935
#> mean(tescore)
#[1] 0.7418301
 