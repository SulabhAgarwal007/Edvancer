library(dplyr)
library(tidyr)
library(stringr)
library(caret)
library(pROC)


setwd("C:/Users/Sulabh/Documents/Edvancer Data Science/R Code/Assignment_Retail")
# Read train and test set
train_df = read.csv("store_train.csv", stringsAsFactors = T)
test_df = read.csv("store_test.csv", stringsAsFactors = T)

glimpse(train_df)

# Function To check for Nulls in each column
check_missing = function(df){
    for(col in names(df)){
      print(paste0(col," - ",sum(is.na(df[, col]))))
      }
  }
# Function to get data type and unique counts for each column
unique_count = function(df){
for(col in names(df)){
  print(paste0(col," - ",typeof(df[,col])," - ",length(unique(df[, col]))))
  }
}

# To merge train and test into one dataset for pre-processing
train_df$set_type = "train"
test_df$set_type = "test"
test_df$store = NA
df= rbind(train_df[,sort(names(train_df))], test_df[,sort(names(train_df))])



check_missing(df)
unique_count(df)

# Process Countytownname
# function to extract county type from countytownname
x = function(x){
  str_sub(x, str_locate_all(x, " ")[[1]][dim(str_locate_all(x, " ")[[1]])[1]],-1) %>%
    str_replace("City","city") %>% str_replace("city\\)","city")
  }

df$total_sales = df$sales0+df$sales1+df$sales2+df$sales3+df$sales4

df[,c("sales0","sales1","sales2","sales3","sales4")] = NULL

df$countytownname_sep = sapply(df$countytownname,x)%>%trimws()

df[!df$countytownname_sep %in% c("city","town","County") , "countytownname_sep"] = "others"

#Process Storecode to extract first 6 characters
df$storecode = str_sub(df$storecode,0,6)

unique_count(df)

#removing redundant columns
df[, c("Id","Areaname","countyname","countytownname","state")] = NULL

# convert processed character columns as factor 
for(col in names(df)){
  if(typeof(df[,col])=="character"){
    df[, col] = as.factor(df[,col])
  }
}

# with this dataset seems to be ready to be trained
# glimpse(df)

processModel = preProcess(df[,!names(df) %in% ("store")], 
                          method = c( "knnImpute")) # "center", "scale",

df = predict(processModel, newdata = df)

#Dummy Variable creation using one hot encoding
dmy = dummyVars(~ .-set_type, df, sep="_", fullRank = TRUE)
df = data.frame(predict(dmy, newdata = df))

# df[, c("store_Type","storecode","countytownname_sep")] = NULL

#Check missing values for processing
check_missing(df)



# Test near zero variance variable
nzv = nearZeroVar(df)
nzv

df = df[, -nzv]
# seperate out test and train dataset, but set_type column got removed while creating dummy variables
train_df = df[!(is.na(df$store)),]
test_df = df[is.na(df$store),]

train_df$set_type = NULL
test_df$set_type = NULL

# Create partition in Train Dataset for validation
train_index = createDataPartition(train_df$store, p=0.7, list=FALSE )

train_df_split_train = train_df[train_index, ]
train_df_split_test = train_df[-train_index,]

# define outcome and predictor variable names
outcomeName = 'store'
train_df_split_train[, outcomeName] = as.factor(train_df_split_train[,outcomeName])
train_df_split_test[, outcomeName] = as.factor(train_df_split_test[, outcomeName])

predictors = names(train_df_split_train)[!names(train_df_split_train) %in% outcomeName]

# this code is to perform recursive feature extraction
control = rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE,
                      number = 5)

loadprofile = rfe(train_df_split_train[,predictors], train_df_split_train[,outcomeName], 
                   rfeControl = control)

opt_var = loadprofile$optVariables

opt_var = loadprofile$optVariables[!(loadprofile$optVariables %in% 
                                       c("CouSub", "population","countytownname_sep_County",
                                         "state_alpha_ME","total_sales"))]

# top_5 = c("storecode_METRO4", "storecode_METRO3", "storecode_METRO2", "storecode_NCNTY3", "sales0")



fit_control = trainControl(method = "repeatedcv", repeats = 5, number = 5)

model_glm = train(train_df_split_train[,opt_var], train_df_split_train[, outcomeName],
                  method = 'glm',
                  trControl = fit_control,
                  tuneLength = 10
                  )

# Variable importance
varImp(object = model_glm)

model_pred = predict(model_glm, train_df_split_test[,predictors], type = "raw")

confusionMatrix(model_pred, train_df_split_test[,outcomeName])

model_glm

levels(train_df_split_train$store) = c("Y","N")
levels(train_df_split_test$store) = c("Y","N")

rf_fit_control = trainControl(method = "cv", number = 5, 
                              savePredictions = TRUE, 
                              classProbs = TRUE, 
                              verboseIter = TRUE)

rf_grid <- expand.grid(mtry = c(2, 3, 4, 5),
                      splitrule = c("gini", "extratrees"),
                      min.node.size = c(1, 3, 5))

rf_fit <- train(train_df_split_train[,opt_var], train_df_split_train[, outcomeName], 
                method = "ranger",
                trControl = rf_fit_control,
                metric = 'ROC',
                # provide a grid of parameters
                # tuneGrid = rf_grid
                tuneLength = 10
                )

rf_fit

model_rf_pred = predict(rf_fit, train_df_split_test[,predictors], type= 'prob')[,2]

confusionMatrix(model_rf_pred, train_df_split_test[,outcomeName])

temp <- factor(train_df_split_test$store, ordered = T, levels = c("Y","N"))

auc_roc(model_rf_pred, temp)

p = predict(rf_fit, test_df, type = "prob")[,2]

write.csv(p,'Sulabh_Agarwal_P2_part2.csv',row.names=F)

######################Below can be ignored.
# gbm

# Set up training control
ctrl <- trainControl(method = "repeatedcv",   # 10fold cross validation
                     number = 5,							# do 5 repititions of cv
                     summaryFunction=twoClassSummary,	# Use AUC to pick the best model
                     classProbs=TRUE,
                     allowParallel = TRUE)

grid <- expand.grid(interaction.depth=c(1,2), # Depth of variable interactions
                    n.trees=c(10,20),	        # Num trees to fit
                    shrinkage=c(0.01,0.1),		# Try 2 values for learning rate 
                    n.minobsinnode = 20)
#											
set.seed(1951)  # set the seed




gbm.tune <- train(x=train_df_split_train[,opt_var],y=train_df_split_train$store,
                              method = "gbm",
                              metric = "ROC",
                              trControl = ctrl,
                              tuneLength = 10,
                              #tuneGrid=grid,
                              verbose=FALSE)

model_gbm_pred = predict(gbm.tune, train_df_split_test[,predictors])

confusionMatrix(model_gbm_pred, train_df_split_test[,outcomeName])


