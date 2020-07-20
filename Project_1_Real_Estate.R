library(dplyr)
library(tidyr)
library(shiny)
library(caret)
# library(doParallel)

# C1 = makePSOCKcluster(3)
# registerDoParallel(C1)

setwd("C:/Users/Sulabh/Documents/Edvancer Data Science/R Code/Assignment_Housing")

train_df = read.csv("housing_train.csv", stringsAsFactors = FALSE)
# head(train_df)

test_df = read.csv("housing_test.csv", stringsAsFactors = FALSE)
# head(test_df)

sum(which(is.null(train_df$Price))) # Check if price is Null in train dataset

train_df$dataset = "train"
test_df$dataset = "test"
test_df$Price = "NA"

col_names = sort(names(train_df))

combined_df = rbind(train_df[, col_names], test_df[, col_names])

#Seperate out council area Null

# v_null_CA = which(combined_df$CouncilArea=="")

# combined_df1 = combined_df[-v_null_CA,]
# combined_df2 = combined_df[v_null_CA,]

combined_df1 = combined_df # to by-pass code for split on Null CouncilArea
# combined_df1[is.null(combined_df1$CouncilArea),] =

replace(combined_df1$CouncilArea, which(combined_df1$CouncilArea == ""), "U")  
  
for(col in col_names){
  # sprinf never prints any value on screen, it just returns string type values.
  print(sprintf("%s %d", col, sum(is.na(combined_df1[, col]))))
  # print(table(combined_df1[,col]))
}

v_currentYear = as.integer(format(Sys.Date(),"%Y"))


# combined_df1$Address = trimws(combined_df1$Address)

# combined_df1 = separate(combined_df1, "Address", c("House No","Loc","Rd_St"), sep=" ")  

combined_df1 <- combined_df1 %>% 
  mutate(postcode_cat = case_when(
    Postcode>=3000 & Postcode< 3025 ~ "cat A",
    Postcode>=3025 & Postcode< 3050 ~ "Cat B",
    Postcode>=3050 & Postcode< 3075 ~ "cat C",
    Postcode>=3075 & Postcode< 3100 ~ "cat D",
    Postcode>=3100 & Postcode< 3125 ~ "Cat E",
    Postcode>=3125 & Postcode< 3150 ~ "cat F",
    Postcode>=3150 & Postcode< 3175 ~ "cat G",
    Postcode>=3175  ~ "cat H"
    )
  )

combined_df1[, c("Address","Postcode", "Suburb","SellerG")] = NULL

combined_df1$CouncilArea = as.factor(combined_df1$CouncilArea)
combined_df1$Method = as.factor(combined_df1$Method)
combined_df1$Type = as.factor(combined_df1$Type)
combined_df1$postcode_cat = as.factor(combined_df1$postcode_cat)


dmy = dummyVars( Price ~  CouncilArea + Method + Type + postcode_cat, 
                 data = combined_df1, sep = "_",
                 fullRank = TRUE
                 )

combined_df1 = cbind(combined_df1, predict(dmy, newdata = combined_df1))

combined_df1[, c("CouncilArea", "Method", "Type", "postcode_cat")] = NULL

names(combined_df1) = gsub(" ","",names(combined_df1))

str(combined_df1)

pp_model = preProcess(combined_df1[,  !names(combined_df1) %in% c("dataset", "Price")], 
                      method = c("bagImpute","center", "scale","nzv"))

combined_df1 = predict(pp_model, newdata = combined_df1)

combined_df1$Price = as.numeric(combined_df1$Price)
combined_df1$YearBuilt = v_currentYear - floor(combined_df1$YearBuilt)



train_df = combined_df1 %>% filter(dataset == "train") %>% select(-"dataset")
test_df = combined_df1 %>% filter(dataset == "test") %>% select(-c("dataset","Price"))

# test train split

train_index = createDataPartition(train_df$Price, p=0.7, list=FALSE )

train_df_split_train = train_df[train_index, ]
train_df_split_test = train_df[-train_index,]

fit_control = trainControl(method = "repeatedcv", repeats = 5, number = 10)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5), 
                        n.trees = (1:10)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

model = train(Price~., 
              data=train_df_split_train,
              method = 'gbm',
              trControl = fit_control,
              tuneGrid = gbmGrid,
              verbose = FALSE
              )

model

# stopCluster(C1)

p = predict(model, train_df_split_test)

v_RMSE = RMSE(p, train_df_split_test$Price)

Score = 212467/v_RMSE
Score

p = predict(model, test_df)
write.csv(p,'Sulabh_Agarwal_P1_part2.csv',row.names=F)


