load('~/Downloads/data_hw3_deepsolar.RData')
library(ROCR)

# Supervised Learning Methods

# 1. Logistic Regression. 
fit <- glm(solar_system_coverage ~ ., data = data, family = "binomial")

summary(fit)

tau <- 0.5
p <- fitted(fit)
pred <- ifelse(p > tau, 'low', 'high')
a <- fit$model

# cross tabulation between observed and predicted
table(data$solar_system_coverage, pred)

# compute accuracy for given tau
tab <- table(data$solar_system_coverage, pred)
tab[1, 1]/ (tab[1, 1] + tab[1, 2])
tab[2, 2]/ (tab[2, 2] + tab[2, 1])
sum(diag(tab))/sum(tab)

# finding optimal tau
pred_obj <- prediction(fitted(fit), data$solar_system_coverage)

roc <- performance(pred_obj, "tpr", "fpr")
plot(roc)
abline(0, 1, col = "darkorange2", lty = 2) # add bisect line

# compute the area under the ROC curve
auc <- performance(pred_obj, "auc")
auc@y.values

sens <- performance(pred_obj, "sens")
spec <- performance(pred_obj, "spec")
tau <- sens@x.values[[1]]
sens_spec <- sens@y.values[[1]] + spec@y.values[[1]]
best_roc <- which.max(sens_spec)
plot(tau, sens_spec, type = "l")
points(tau[best_roc], sens_spec[best_roc], pch = 19, col = adjustcolor("darkorange2", 0.5))
tau[best_roc]

pred <- ifelse(fitted(fit) > tau[best_roc], 'low', 'high')
table(data$solar_system_coverage, pred)

# accuracy for optimal tau
acc <- performance(pred_obj, "acc")
acc@y.values[[1]][best_roc]

# sensitivity and specificity for optimal tau
sens@y.values[[1]][best_roc]
spec@y.values[[1]][best_roc]


# 2. Bagging and random forests
N <- nrow(data)
N_train <- sample(1:nrow(data), N*0.8)
N_test <- setdiff(1:N, train)

library(randomForest)

# implement the random forest algorithm
fit_rf <- randomForest(solar_system_coverage ~ ., data = data, subset = N_train, importance = TRUE)

probs <- predict(fit_rf, type = "prob", newdata = data[N_test,])
head(probs)

# check predictions
pred_rf <- predict(fit_rf, type = "class", newdata = data[test,])
table(data$solar_system_coverage[test], pred_rf)

tab <- table(data$solar_system_coverage[test], pred_rf)
tab[1, 1]/ (tab[1, 1] + tab[1, 2])
tab[2, 2]/ (tab[2, 2] + tab[2, 1])
sum(diag(tab))/sum(tab)


help("randomForest")

# 3. Support Vector Machines.

library(kernlab)

x <- data[, c(-1)]
y <- data[, c(1)]

y <-ifelse(y == 'high', 1, 0)

# set aside original data (just in case...)
x0 <- x
y0 <- y
# set aside test data
N <- nrow(x)
test <- sample(1:N, N*0.2)
x_test <- x[test,]
y_test <- y[test]


train <- setdiff(1:N, test)
x <- x[train,]
y <- y[train]
N_train <- nrow(x)


# we use this function to compute classification accuracy
class_acc <- function(y, yhat) {
  tab <- table(y, yhat)
  return( sum(diag(tab))/sum(tab) )
}

C <- c(1, 2, 5, 10, 20)
sigma <- c(0.010, 0.015, 0.020, 0.025, 0.030)
grid <- expand.grid(C, sigma)
colnames(grid) <- c("C", "sigma")

# check
# head(grid, 10)

# total size of the grid
n_mod <- nrow(grid)

K <- 4 # set number of folds
R <- 4 # set number of replicates --- NOTE : could be slow

out <- vector("list", R) # store accuracy output
# out is a list, each slot of this list will contain a matrix where each column
# corresponds to the accuracy of each AVM classifier in the K folds


for ( r in 1:R ) {
  acc <- matrix(NA, K, n_mod) # accuracy of the classifiers in the K folds
  folds <- rep(1:K, ceiling(N_train/K))
  folds <- sample(folds) # random permute
  folds <- folds[1:N_train] # ensure we got N_train data points
  for ( k in 1:K ) {
    train_fold <- which(folds != k)
    validation <- setdiff(1:N_train, train_fold)
    # fit SVM on the training data and assess on the validation set
    for ( j in 1:n_mod ) {
      fit <- ksvm(data.matrix(x[train_fold,]), y[train_fold], type = "C-svc", kernel = "rbfdot", C = grid$C[j], kpar = list(sigma = grid$sigma[j]))
      pred <- predict(fit, newdata = x[validation,])
      acc[k,j] <- class_acc(pred, y[validation])
    }
  }
  
  out[[r]] <- acc
  # print(r) # print iteration number
}

out[[4]]

avg_fold_acc <- t( sapply(out, colMeans) )
head(avg_fold_acc, 3)

avg_acc <- colMeans(avg_fold_acc) # estimated mean accuracy
grid_acc <- cbind(grid, avg_acc)
grid_acc

best <- which.max(grid_acc$avg_acc)
grid_acc[best,]