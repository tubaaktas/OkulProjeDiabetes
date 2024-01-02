from research import *

data = load_data("database/diabetes.csv")

# Kolon isimlerini büyültüyorum ki sorgulama vs yaparken yazmak, okumak kolay olsun.
data.columns = [col.upper() for col in data.columns]
data.head()

check_df(data)

plot_outliers(data)

plot_correlation_heatmap(data)

y = data["OUTCOME"]
X = data.drop("OUTCOME", axis=1)

model_results = base_models_results(X,y)

# Her bir performans metriği için grafik oluşturma
metrics = list(model_results[list(model_results.keys())[0]].keys())
for metric in metrics:
    plot_metric(metric, model_results, title="base model için")


cat_cols, num_cols, cat_but_car = grab_col_names(data)


for col in num_cols:
    target_summary_with_num(data, "OUTCOME", col)


for col in num_cols:
    num_summary(data, col, plot=False)


cat_summary(data, "OUTCOME")

#0 değerlerini NaN olarak değiştirelim.
nan_col=['GLUCOSE','BLOODPRESSURE','SKINTHICKNESS', 'INSULIN', 'BMI']
data = fill_zeros_with_nan(data,nan_col)
#Boş değerleri dolduralım
data = fill_missing_values(data)

for col in num_cols:
  print(col,"için LOW LIMIT, UP LIMIT değerleri = ", outlier_thresholds(data,col),"\n")

for col in num_cols:
  print(col,"için outlier kontrolü", check_outlier(data,col),"\n")

for col in num_cols:
  print(col,"için var olan outlierlar\n", grab_outliers(data,col))


#şimdi kalan aykırı değerlerin değişimini sağlayabiliriz.
for col in num_cols:
  print(replace_with_thresholds(data, col))

plot_histogram(data, bins=20)

X, y = data_process(data)

print(data.isnull().sum())

model_results = base_models_results(X, y)


# Her bir performans metriği için grafik oluşturma (Feature Engineering Sonrası)
metrics = list(model_results[list(model_results.keys())[0]].keys())
for metric in metrics:
    plot_metric(metric, model_results, title="Feature Engineering Sonrası")


################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean() #0.8959086664969019
cv_results['test_f1'].mean()#0.8485229241795583
cv_results['test_roc_auc'].mean()#0.949535290006988

#Hiperparametre optimizasyonu
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()#0.8998047703930057
cv_results['test_f1'].mean()#0.8519878866133057
cv_results['test_roc_auc'].mean()#0.9546701607267645


################################################
# XGBoost
################################################

xgb_model = XGBClassifier(random_state=17)
xgb_model.get_params()
cv_results = cross_validate(xgb_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

#XGBoost için hiperparametre optimizasyonu
cv_results['test_accuracy'].mean()#0.8933197521432816
cv_results['test_f1'].mean()#0.8460027437634992
cv_results['test_roc_auc'].mean()#9500230607966458


xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgb_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgb_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()#0.8945929887106358
cv_results['test_f1'].mean()#0.846351262053318
cv_results['test_roc_auc'].mean()#0.9535590496156534


################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}


rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()#0.8906698564593303
cv_results['test_f1'].mean()#0.8378046491808121
cv_results['test_roc_auc'].mean()#0.9507492877492878



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# VotingClassifier oluşturma
voting_clf = VotingClassifier(estimators=[('lgbm', lgbm_final), ('xgb', xgboost_final),('rf', rf_final)], voting='soft')

# Modeli eğitme
voting_clf.fit(X_train, y_train)
xgboost_final.fit(X_train,y_train)
lgbm_final.fit(X_train,y_train)
rf_final.fit(X_train,y_train)
# Tahmin yapma
predictions = voting_clf.predict(X_test)
predictions_xgb= xgboost_final.predict(X_test)
predictions_lgbm= lgbm_final.predict(X_test)
predictions_rf= rf_final.predict(X_test)

#Model sonuçları
voting_accuracy, voting_f1, voting_roc_auc = evaluate_model(voting_clf, X, y)
print(f"Voting Classifier - Accuracy: {voting_accuracy}, F1 Score: {voting_f1}, ROC AUC: {voting_roc_auc}")

voting_accuracy, voting_f1, voting_roc_auc = evaluate_model(xgboost_final, X, y)
print(f"XGBoost - Accuracy: {voting_accuracy}, F1 Score: {voting_f1}, ROC AUC: {voting_roc_auc}")

voting_accuracy, voting_f1, voting_roc_auc = evaluate_model(lgbm_final, X, y)
print(f"LGBM - Accuracy: {voting_accuracy}, F1 Score: {voting_f1}, ROC AUC: {voting_roc_auc}")

voting_accuracy, voting_f1, voting_roc_auc = evaluate_model(rf_final, X, y)
print(f"Random Forest - Accuracy: {voting_accuracy}, F1 Score: {voting_f1}, ROC AUC: {voting_roc_auc}")

# Confusion matrix oluşturma
plot_confusion_matrix(y_test,predictions, "voting_clf")
plot_confusion_matrix(y_test,predictions_xgb,"xgb")
plot_confusion_matrix(y_test,predictions_lgbm,"lgbm")
plot_confusion_matrix(y_test,predictions_rf,"rf")

#Önemli değişkenler
plot_importance(xgboost_final,X,name="xgb_imp")
plot_importance(lgbm_final,X,name="lgbm_imp")
plot_importance(rf_final,X,name="rf_imp")

# Modeli kaydetme
joblib.dump(voting_clf, 'voting_model(lgbm_xgb_rf).pkl')