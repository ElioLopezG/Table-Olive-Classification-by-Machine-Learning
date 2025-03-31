if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if("ggplot2" %in% row.names(installed.packages())  == FALSE) install.packages("ggplot2")
if("randomForest" %in% row.names(installed.packages())  == FALSE) install.packages("randomForest")
if("irr" %in% row.names(installed.packages())  == FALSE) install.packages("irr")
if("caret" %in% row.names(installed.packages())  == FALSE) install.packages("caret")
if("reshape2" %in% row.names(installed.packages())  == FALSE) install.packages("reshape2")
if("officer" %in% row.names(installed.packages()) == FALSE) install.packages("officer")
if("rpart" %in% row.names(installed.packages()) == FALSE) install.packages("officer")
if("rpart.plot" %in% row.names(installed.packages()) == FALSE) install.packages("rpart.plot")
if("evtree" %in% row.names(installed.packages()) == FALSE) install.packages("evtree")
if("irr" %in% row.names(installed.packages()) == FALSE) install.packages("irr")
if("ipred" %in% row.names(installed.packages()) == FALSE) install.packages("ipred")
if("gbm" %in% row.names(installed.packages()) == FALSE) install.packages("gbm")
if("xgboost" %in% row.names(installed.packages()) == FALSE) install.packages("xgboost")
if("Biostrings" %in% row.names(installed.packages()) == FALSE) BiocManager::install("Biostrings")

#library
library(ggplot2, quietly = TRUE)
library(randomForest, quietly = TRUE)
library(irr, quietly = TRUE)
library(caret, quietly = TRUE)
library(reshape2, quietly = TRUE)
library(officer, quietly = TRUE)
library(rpart, quietly = TRUE)
library(rpart.plot, quietly = TRUE) 
library(evtree, quietly = TRUE)
library(irr, quietly = TRUE)
library(ipred, quietly = TRUE)
library(gbm, quietly = TRUE)
library(xgboost, quietly = TRUE)
library(Biostrings, quietly = TRUE)

# prepare and read the database
setwd("work directory adress")
database <- read.table(file="database_file",header=T)
row.names(database)<-database[,1] 
database <- subset(database, select=-Sample_name)
# Depending on the target variable to be classified, the other target variables that are not to be classified are removed to prevent them from influencing the classificationPred.database <- database
Pred.database <- subset(Pred.database, select= -Cultivar)
Pred.database <- subset(Pred.database, select= -Country)
Pred.database <- subset(Pred.database, select= -Matrix)
# Convert the target variable to a factor.
Pred.database$Processing_type<-as.factor(Pred.database$Processing_type) 
data<-Pred.database
RFdata<-Pred.database
XGBdata<-Pred.database
# The process starts with the Classification and Regression Trees algorithm

# First, an analysis is carried out to determine the optimal cp
fitrp <- rpart(Processing_type ~ .,data=data)
nsplit<-fitrp$cptable[which.min(fitrp$cptable[,"xerror"]),"nsplit"]; nsplit 
xerror<-min(fitrp$cptable[,"xerror"]); round(xerror, 5)
printcp(fitrp)
xerror_min <- min(fitrp$cptable[,"xerror"])
xerror_se <- fitrp$cptable[which.min(fitrp$cptable[,"xerror"]), "xstd"]
cpopt_index <- which(fitrp$cptable[,"xerror"] <= xerror_min + xerror_se)[1]
cpopt <- fitrp$cptable[cpopt_index, "CP"]

# Lists are created to store the metrics
accuracies_cart <- c()
kappas_cart <- c()
precisions_cart <- c()
recalls_cart <- c()
f1_scores_cart <- c()

# The classification is repeated 10 times in a for loop, changing the seed
n_reps <- 10

for (i in 1:n_reps) {
  set.seed(i)
  
  # Split the dataset into training and testing sets
  trainIndex <- createDataPartition(data$Processing_type, p = 0.8, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]
  
  # Define the cross-validation control
  ctrl <- trainControl(method = "cv", number = 10)
  tuneGrid <- expand.grid(cp = cpopt)
  
  # Train and evaluate the mode
  fit <- train(Processing_type ~ ., data = trainData, method = "rpart", trControl = ctrl, tuneGrid = tuneGrid)
  print(fit)
  
  # Make predictions on the test set
  predictions <- predict(fit, newdata = testData)
  
  # Generate the confusion matrix
  conf_matrix <- confusionMatrix(predictions, testData$Processing_type)
  print(conf_matrix)
  
  # Extract the metrics
  accuracy <- conf_matrix$overall['Accuracy']
  kappa <- conf_matrix$overall['Kappa']
  precision <- mean(conf_matrix$byClass[, 'Precision'], na.rm = TRUE)
  recall <- mean(conf_matrix$byClass[, 'Recall'], na.rm = TRUE)
  f1 <- mean(conf_matrix$byClass[, 'F1'], na.rm = TRUE)
  
  # Store the metrics in the corresponding lists
  accuracies_cart <- c(accuracies_cart, accuracy)
  kappas_cart <- c(kappas_cart, kappa)
  precisions_cart <- c(precisions_cart, precision)
  recalls_cart <- c(recalls_cart, recall)
  f1_scores_cart <- c(f1_scores_cart, f1)
}

# Calculate the means and standard deviations of the metrics
mean_accuracy_cart <- mean(accuracies_cart)
sd_accuracy_cart <- sd(accuracies_cart)
mean_kappa_cart <- mean(kappas_cart)
sd_kappa_cart <- sd(kappas_cart)
mean_precision_cart <- mean(precisions_cart)
sd_precision_cart <- sd(precisions_cart)
mean_recall_cart <- mean(recalls_cart)
sd_recall_cart <- sd(recalls_cart)
mean_f1_cart <- mean(f1_scores_cart)
sd_f1_cart <- sd(f1_scores_cart)

# Print the metrics for classification using CART
cat("Metrics over", n_reps, "runs:\n")
cat("Accuracy: Mean =", mean_accuracy_cart, "SD =", sd_accuracy_cart, "\n")
cat("Kappa: Mean =", mean_kappa_cart, "SD =", sd_kappa_cart, "\n")
cat("Precision: Mean =", mean_precision_cart, "SD =", sd_precision_cart, "\n")
cat("Recall: Mean =", mean_recall_cart, "SD =", sd_recall_cart, "\n")
cat("F1 Score: Mean =", mean_f1_cart, "SD =", sd_f1_cart, "\n")

# Continue with the Random Forest algorithm
# Lists are created to store the metrics
accuracies_rf <- c()
kappas_rf <- c()
precisions_rf <- c()
recalls_rf <- c()
f1_scores_rf <- c()

# The classification is repeated 10 times in a for loop, changing the seed
n_reps <- 10

for (i in 1:n_reps) {
  set.seed(i)
  
  # Split the dataset into training and testing sets
  trainIndex <- createDataPartition(RFdata$processing_type, p = 0.7, list = FALSE)
  trainData <- RFdata[trainIndex, ]
  testData <- RFdata[-trainIndex, ]
  
  # Define the cross-validation control
  ctrl <- trainControl(method = "cv", number = 10)
  
  # Specify a grid of values for mtry
  tuneGrid <- expand.grid(mtry = c(1, 75, 150, 225, 300))
  
  # Train the model with ntree adjustment
  fit <- train(Origin ~ ., data = trainData, method = "rf", trControl = ctrl, tuneGrid = tuneGrid)
  
  # Make predictions on the test set
  predictions <- predict(fit, newdata = testData)
  
  # Generate the confusion matrix
  conf_matrix <- confusionMatrix(predictions, testData$Origin)
  print(conf_matrix)
  
  # Extract the metrics
  accuracy <- conf_matrix$overall['Accuracy']
  kappa <- conf_matrix$overall['Kappa']
  precision_per_class <- mean(conf_matrix$byClass[,'Pos Pred Value'], na.rm = TRUE)
  recall_per_class <- mean(conf_matrix$byClass[,'Sensitivity'], na.rm = TRUE)
  f1_per_class <- 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
  
  # Store the metrics in the corresponding lists
  accuracies_rf <- c(accuracies_rf, accuracy)
  kappas_rf <- c(kappas_rf, kappa)
  precisions_rf <- c(precisions_rf, precision_per_class)
  recalls_rf <- c(recalls_rf, recall_per_class)
  f1_scores_rf <- c(f1_scores_rf, f1_per_class)
  
  importance_rf <- varImp(fit$finalModel)
  variable_importance_list[[i]] <- importance_rf
}

# Calculate the means and standard deviations of the metrics
mean_accuracy_rf <- mean(accuracies_rf)
sd_accuracy_rf <- sd(accuracies_rf)
mean_kappa_rf <- mean(kappas_rf)
sd_kappa_rf <- sd(kappas_rf)
mean_precision_rf <- mean(precisions_rf)
sd_precision_rf <- sd(precisions_rf)
mean_recall_rf <- mean(recalls_rf)
sd_recall_rf <- sd(recalls_rf)
mean_f1_rf <- mean(f1_scores_rf)
sd_f1_rf <- sd(f1_scores_rf)

# Print the metrics for classification using RF
cat("Metrics over", n_reps, "runs:\n")
cat("Accuracy: Mean =", mean_accuracy_rf, "SD =", sd_accuracy_rf, "\n")
cat("Kappa: Mean =", mean_kappa_rf, "SD =", sd_kappa_rf, "\n")
cat("Precision: Mean =", mean_precision_rf, "SD =", sd_precision_rf, "\n")
cat("Recall: Mean =", mean_recall_rf, "SD =", sd_recall_rf, "\n")
cat("F1 Score: Mean =", mean_f1_rf, "SD =", sd_f1_rf, "\n")

# Continue with the eXtreme Gradient Boost algorithm
# Lists are created to store the metrics
accuracies_xgb <- c()
kappas_xgb <- c()
precisions_xgb <- c()
recalls_xgb <- c()
f1_scores_xgb <- c()

# Transform the target variable to numeric and separate the predictor variables into a matrix and the target variable into a vector
data$Processing_type <- as.numeric(data$Processing_type)-1
datax <- as.matrix(subset(data, select=-Processing_type))
datay <- data$Processing_type 
noc <- length(unique(datay)); noc 
# Choose a range to optimize the hyperparameters
grid <- expand.grid(eta = c(1e-2,1e-1, 1, 5, 10), 
                    max_depth = c(1, 3, 7, 10), 
                    gamma = c(0, 2, 5, 10, 20, 100),
                    lambda = c(0, 1e-1, 1, 100, 10000),
                    alpha = c(0, 1e-1, 1, 100, 10000),
                    kappa = 0,
                    optntrees = 0, 
                    mlogloss = 0)
# Run a classification testing different hyperparameters
for(i in 1:nc) {
  
  fitxgb <- xgb.cv(data = trainx,label = trainy, nrounds = 500, nthread = 2, 
                   metrics = "mlogloss", early_stopping_rounds = 10, nfold = 5,verbose = F,
                   subsample = .8, colsample_bytree = .8,objective = "multi:softmax",
                   prediction = T,
                   eta = grid$eta[i],
                   max_depth = grid$max_depth[i],
                   gamma = grid$gamma[i], 
                   lambda = grid$lambda[i], 
                   alpha = grid$alpha[i],
                   num_class = noc)
  
  probs = as.data.frame(fitxgb$pred)[,1] 
  pred <- round(probs,0)
  traindata$pred<-pred 
  k<-kappa2(traindata[,c(1,443)], "equal")
  
  grid$kappa[i] <- round(k$value, 3)  
  grid$optntrees[i] <- which.min(fitxgb$evaluation_log$test_mlogloss_mean)
  grid$minRMSE[i] <- min(fitxgb$evaluation_log$test_mlogloss_mean)
  print(i)
}
# The best result with a higher kappa is chosen as the selected hyperparameters
grid<-grid[order(grid$kappa,decreasing=T),]
head(grid, 5)


# The classification is repeated 10 times in a for loop, changing the seed
n_reps <- 10

for (i in 1:n_reps) {
  set.seed(i)
  
  # Split the dataset into training and testing sets
  s <- sample.int(n = nrow(XGBdata), size = floor(.8 * nrow(XGBdata)), replace = FALSE)
  traindata <- XGBdata[s, ]
  testdata <- XGBdata[-s, ]
  
  # Separate the predictors from the target variable and transform them
  trainx <- as.matrix(subset(traindata, select = -Processing_type))
  trainy <- traindata$Processing_type
  testx <- as.matrix(subset(testdata, select = -Processing_type))
  testy <- testdata$Processing_type
  
  # Transform the data into the appropriate structure for xgboost
  xgb_train <- xgb.DMatrix(data = trainx, label = trainy)
  xgb_test <- xgb.DMatrix(data = testx, label = testy)
  
  # Train the model using the hyperparameters optimized previously
  fitxgbcv <- xgb.cv(
    objective = "multi:softmax",
    eval_metric = "mlogloss",
    early_stopping_rounds = 10,
    num_class = length(unique(data$Processing_type)),
    nfold = 10,
    nrounds = 500,
    verbose = FALSE,
    data = xgb_train,
    prediction = TRUE,
    eta = 0.1,
    max_depth = 7,
    gamma = 0,
    lambda = 1,
    alpha = 1
  )
  
  # Obtain the predictions from the best model
  best_ntree <- fitxgbcv$best_iteration
  xgb_model <- xgboost(data = xgb_train, objective = "multi:softmax", num_class = length(unique(data$Processing_type)),
                       nrounds = best_ntree, eta = 0.1, max_depth = 7, gamma = 0, lambda = 1, alpha = 1, verbose = FALSE)
  predictions <- predict(xgb_model, xgb_test)
  
  # Generate the confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(testy))
  print(conf_matrix)
  
  # Extract the metrics
  accuracy <- conf_matrix$overall['Accuracy']
  kappa <- conf_matrix$overall['Kappa']
  precision_per_class <- mean(conf_matrix$byClass[,'Pos Pred Value'], na.rm = TRUE)
  recall_per_class <- mean(conf_matrix$byClass[,'Sensitivity'], na.rm = TRUE)
  f1_per_class <- 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
  
  # Store the metrics in the corresponding lists
  accuracies_xgb <- c(accuracies_xgb, accuracy)
  kappas_xgb <- c(kappas_xgb, kappa)
  precisions_xgb <- c(precisions_xgb, precision_per_class)
  recalls_xgb <- c(recalls_xgb, recall_per_class)
  f1_scores_xgb <- c(f1_scores_xgb, f1_per_class)
}

# Calculate the means and standard deviations of the metrics
mean_accuracy_xgb <- mean(accuracies_xgb)
sd_accuracy_xgb <- sd(accuracies_xgb)
mean_kappa_xgb <- mean(kappas_xgb)
sd_kappa_xgb <- sd(kappas_xgb)
mean_precision_xgb <- mean(precisions_xgb)
sd_precision_xgb <- sd(precisions_xgb)
mean_recall_xgb <- mean(recalls_xgb)
sd_recall_xgb <- sd(recalls_xgb)
mean_f1_xgb <- mean(f1_scores_xgb)
sd_f1_xgb <- sd(f1_scores_xgb)

# Print the metrics for classification using XGB
cat("Metrics over", n_reps, "runs:\n")
cat("Accuracy: Mean =", mean_accuracy_xgb, "SD =", sd_accuracy_xgb, "\n")
cat("Kappa: Mean =", mean_kappa_xgb, "SD =", sd_kappa_xgb, "\n")
cat("Precision: Mean =", mean_precision_xgb, "SD =", sd_precision_xgb, "\n")
cat("Recall: Mean =", mean_recall_xgb, "SD =", sd_recall_xgb, "\n")
cat("F1 Score: Mean =", mean_f1_xgb, "SD =", sd_f1_xgb, "\n")