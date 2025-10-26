# =========================================
# Random Forest Regression on EV Dataset (2025)
# =========================================

# 1. Load required packages
if(!require(caret)) install.packages("caret", dependencies = TRUE)
if(!require(randomForest)) install.packages("randomForest", dependencies = TRUE)
library(caret)
library(randomForest)

# 2. Load the dataset
data <- read.csv("C:/Users/yoges/OneDrive/Desktop/electric_vehicles_spec_2025.csv.csv", stringsAsFactors = FALSE)

# 3. Replace "?" and empty strings with NA
data[data == "?"] <- NA
data[data == ""] <- NA

# 4. Drop columns with mostly missing values or irrelevant for modeling
drop_cols <- c("source_url", "model", "brand")  # drop URLs and identifiers
data <- data[, !(names(data) %in% drop_cols)]

# 5. Convert numeric columns
numeric_cols <- c("top_speed_kmh", "battery_capacity_kWh", "number_of_cells", "torque_nm",
                  "efficiency_wh_per_km", "range_km", "acceleration_0_100_s",
                  "fast_charging_power_kw_dc", "towing_capacity_kg", "cargo_volume_l",
                  "seats", "length_mm", "width_mm", "height_mm")

for(col in numeric_cols){
  if(col %in% names(data)){
    data[[col]] <- as.numeric(data[[col]])
    if(!all(is.na(data[[col]]))){
      data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
    } else {
      data[[col]] <- 0
    }
  }
}

# 6. Convert categorical columns to factors
categorical_cols <- setdiff(names(data), numeric_cols)
for(col in categorical_cols){
  data[[col]] <- as.factor(data[[col]])
}

# 7. Remove rows with missing target variable
data <- data[!is.na(data$range_km), ]

# 8. Drop factor columns with only one level
drop_single_level <- sapply(data, function(x) is.factor(x) && length(unique(x)) < 2)
data <- data[, !drop_single_level]

# 9. Split into training and testing sets
set.seed(123)
train_index <- createDataPartition(data$range_km, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# 10. Remove any remaining NAs
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)

# 11. Train Random Forest with hyperparameter tuning
set.seed(123)
control <- trainControl(method = "cv", number = 5)
tune_grid <- expand.grid(mtry = 2:10)

rf_model <- train(range_km ~ ., 
                  data = train_data,
                  method = "rf",
                  metric = "RMSE",
                  tuneGrid = tune_grid,
                  trControl = control,
                  ntree = 500)

# 12. Best model info
print(rf_model)
cat("Best mtry:", rf_model$bestTune$mtry, "\n")

# 13. Predict on test set
predictions <- predict(rf_model, newdata = test_data)

# 14. Evaluate performance
rmse <- sqrt(mean((predictions - test_data$range_km)^2))
r2 <- cor(predictions, test_data$range_km)^2
cat("RMSE on test data:", rmse, "\n")
cat("R-squared on test data:", r2, "\n")

# 15. Feature importance plot
importance <- varImp(rf_model, scale = TRUE)
plot(importance, top = 15)

# 16. Compare actual vs predicted
results <- data.frame(Actual = test_data$range_km, Predicted = predictions)
head(results)
