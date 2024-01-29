from research import *

data = load_data("database/diabetes.csv")
"""
outcome_0_rows = data[data['Outcome'] == 0]

# Eğer 500'den fazla Outcome değeri 0 olan satır varsa, 250 tanesini rastgele seçip diğerlerini çıkartalım
if len(outcome_0_rows) >= 500:
    random_selected_rows = outcome_0_rows.sample(n=300, random_state=42)
    data_filtered = pd.concat([data[data['Outcome'] == 1], random_selected_rows], ignore_index=True)

    # Sonucu yazdırma
    print(data_filtered)
else:
    print("500'den fazla Outcome değeri 0 olan satır bulunmuyor.")

check_df(data_filtered)
"""
cat_cols, num_cols, cat_but_car = grab_col_names(data)


plot_outliers(data)

plot_correlation_heatmap(data)

y = data["Outcome"]
X = data.drop("Outcome", axis=1)

model_results = base_models_results(X,y)

# Her bir performans metriği için grafik oluşturma
metrics = list(model_results[list(model_results.keys())[0]].keys())
for metric in metrics:
    plot_metric(metric, model_results, title="base model için")


for col in num_cols:
    target_summary_with_num(data, "Outcome", col)


for col in num_cols:
    num_summary(data, col, plot=False)

cat_summary(data, "Outcome")

#0 değerlerini NaN olarak değiştirelim.
nan_col=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
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

X, y, num_cols = data_process(data)
num_cols.iloc[500]
print(data.isnull().sum())

model_results = base_models_results(X, y)



# Her bir performans metriği için grafik oluşturma (Feature Engineering Sonrası)
metrics = list(model_results[list(model_results.keys())[0]].keys())
for metric in metrics:
    plot_metric(metric, model_results, title="Feature Engineering Sonrası")

################################################################################################

# LightGBM

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


################################################################################################

# XGBoost

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


################################################################################################################################################

# Random Forests

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


########################################################################################################################
#MODEL EĞİTİMİ
########################################################################################################################
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


"""
########################################################################################################################

#Önemli değişkenler
plot_importance(xgboost_final, X,name="xgb_imp")
plot_importance(lgbm_final,X,name="lgbm_imp")
plot_importance(rf_final,X,name="rf_imp")


#ÖNEMLİ DEĞİŞKENLERLE MODEL EĞİTİMİ

important_features = ["Insulin", "Glucose", "Age", "BMI", "SkinThickness", "DiabetesPedigreeFunction","NEW_INSULIN_Normal"]
X_important = X[important_features]

voting_clf = VotingClassifier(estimators=[('lgbm', lgbm_final), ('xgb', xgboost_final),('rf', rf_final)], voting='soft')

# Modeli eğitme
voting_clf.fit(X_important, y)
xgboost_final.fit(X_important,y)
lgbm_final.fit(X_important,y)
rf_final.fit(X_important,y)

# Tahmin yapma
predictions = voting_clf.predict(X_important)
predictions_xgb = xgboost_final.predict(X_important)
predictions_lgbm= lgbm_final.predict(X_important)
predictions_rf = rf_final.predict(X_important)

models = [voting_clf, xgboost_final, lgbm_final, rf_final]
model_names = ["Voting Classifier", "XGBoost", "LGBM", "Random Forest"]

for model, name in zip(models, model_names):
    accuracy, f1, roc_auc, conf_matrix = evaluate_model(model, X_important, y)
    print(f"{name} - Accuracy: {accuracy}, F1 Score: {f1}, ROC AUC: {roc_auc}")


X_imp_train, X_imp_test, y_train, y_test = train_test_split(X_important, y, test_size=0.3, random_state=42)
voting_clf = VotingClassifier(estimators=[('lgbm', lgbm_final), ('xgb', xgboost_final),('rf', rf_final)], voting='soft')

# Modeli eğitme
voting_clf.fit(X_imp_train, y_train)
xgboost_final.fit(X_imp_train,y_train)
lgbm_final.fit(X_imp_train,y_train)
rf_final.fit(X_imp_train,y_train)

# Tahmin yapma
predictions = voting_clf.predict(X_imp_test)
predictions_xgb= xgboost_final.predict(X_imp_test)
predictions_lgbm= lgbm_final.predict(X_imp_test)
predictions_rf= rf_final.predict(X_imp_test)

#Model sonuçları
voting_accuracy, voting_f1, voting_roc_auc, _ = evaluate_model(voting_clf, X_important, y)
print(f"Voting Classifier - Accuracy: {voting_accuracy}, F1 Score: {voting_f1}, ROC AUC: {voting_roc_auc}")
print("#####################################################################################")
xgboost_accuracy, xgboost_f1, xgboost_roc_auc, _ = evaluate_model(xgboost_final, X_important, y)
print(f"XGBoost - Accuracy: {xgboost_accuracy}, F1 Score: {xgboost_f1}, ROC AUC: {xgboost_roc_auc}")
print("#####################################################################################")

lgbm_accuracy, lgbm_f1, lgbm_roc_auc, _ = evaluate_model(lgbm_final, X_important, y)
print(f"LGBM - Accuracy: {lgbm_accuracy}, F1 Score: {lgbm_f1}, ROC AUC: {lgbm_roc_auc}")
print("#####################################################################################")

rf_accuracy, rf_f1, rf_roc_auc, _ = evaluate_model(rf_final, X_important, y)
print(f"Random Forest - Accuracy: {rf_accuracy}, F1 Score: {rf_f1}, ROC AUC: {rf_roc_auc}")


# Confusion matrix oluşturma
plot_confusion_matrix(y_test,predictions, "voting_clf")
plot_confusion_matrix(y_test,predictions_xgb,"xgb")
plot_confusion_matrix(y_test,predictions_lgbm,"lgbm")
plot_confusion_matrix(y_test,predictions_rf,"rf")

#Önemli değişkenler
plot_importance(xgboost_final, X_important,name="xgb_imp")
plot_importance(lgbm_final,X_important,name="lgbm_imp")
plot_importance(rf_final,X_important,name="rf_imp")
"""

"""
# Rastgele bir kullanıcı verisi oluşturma
random_user_age = np.random.randint(18, 65)  # Örneğin, 18 ile 65 yaş arasında rastgele bir yaş
random_user_glucose = np.random.randint(70, 200)  # Örneğin, 70 ile 200 arasında rastgele bir glukoz değeri
random_user_insulin = np.random.randint(5, 200)  # Örneğin, 5 ile 200 arasında rastgele bir insülin değeri
random_user_bmi = np.random.uniform(18.5, 35.0)  # Örneğin, 18.5 ile 35.0 arasında rastgele bir BMI değeri

# Veriyi modelin girişine uygun hale getirme
random_user_data = np.array([[random_user_age, random_user_glucose, random_user_insulin, random_user_bmi]])

# Modelden tahmin yapma
prediction = xgboost_final.predict(random_user_data)

# Tahmin sonucunu yazdırma
print(f"Tahmin Sonucu: {prediction}")"""