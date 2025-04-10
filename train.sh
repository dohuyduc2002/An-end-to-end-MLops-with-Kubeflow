# Run training using XGBoost
./underwriting_trainer.py xgb \
  --processed_train "data/processed_train.csv" \
  --processed_test "data/processed_test.csv" \
  --version "v1" \
  --experiment_name "underwriting_xgb" \
  --xgb_max_depth 6 \
  --xgb_learning_rate 0.05 \
  --xgb_n_estimators 200 \
  --xgb_subsample 0.9 \
  --xgb_colsample_bytree 0.9

# ./underwriting_trainer.py lgbm \
#   --processed_train "data/processed_train.csv" \
#   --processed_test "data/processed_test.csv" \
#   --version "v1" \
#   --experiment_name "underwriting_lgbm" \
#   --lgb_max_depth 7 \
#   --lgb_learning_rate 0.05 \
#   --lgb_n_estimators 150 \
#   --lgb_subsample 0.85 \
#   --lgb_colsample_bytree 0.85
