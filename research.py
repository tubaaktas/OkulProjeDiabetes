import joblib
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score, train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scikitplot.metrics import plot_roc_curve
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_data(path):
    data = pd.read_csv(path)
    return data

def check_df(dataframe, head=8):
  print("##### Shape #####") #kac kolon kac satir
  print(dataframe.shape)
  print("##### Types #####") #kolonların, yani özelliklerin tipleri (int,float,str..)
  print(dataframe.dtypes)
  print("##### Tail #####") #Veri setinin son 5 degerini inceliyoruz.
  print(dataframe.tail(head))
  print("##### Head #####") #Veri setinin ilk 5 degerini inceliyoruz.
  print(dataframe.head(head))
  print("##### Null Analysis #####") #Bos deger olup olmadigini kontrol ediyoruz.
  print(dataframe.isnull().sum())
  print("##### Quantiles #####") #sayısal verilere sahip olan sütunların istatiksel değerlerine baktık.
  #Hamilelik max 17 olarak girilmis, 17 kere hamilelik kulaga biraz imkansız geliyor. Verimizde Outlier degerler bulunuyor.
  # Mesela glucose degeri min olarak 0 gosterilmis, glucose 0 olamaz. Demek ki null kısımlara 0 girilmis.
  print(dataframe.describe([0,0.05, 0.50, 0.95, 0.99, 1]).T)

################################################
# GRAFİKLER, GÖSTERİMLER
################################################

#CONFUSION MATRİX İÇİN
def plot_confusion_matrix(y_true, y_pred,model_name="Model Default"):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

#F1, ROC_AUC, ACCURACY VB GİBİ DEĞERLERİN GRAFİKLEŞTİRİLMESİ

def plot_metric(metric, model_results, title):
    models = list(model_results.keys())
    values = [model_results[model][metric] for model in models]

    plt.figure(figsize=(8, 6))
    plt.bar(models, values)
    plt.xlabel('Models')
    plt.ylabel(metric)
    plt.title(f'{metric} Comparison Among Models {title}')
    plt.ylim(0, 1)  # Performans metrikleri genellikle 0 ile 1 arasında olur
    plt.show()

#HİSTOGRAM GRAFİĞİ İÇİN
def plot_histogram(data, bins=10):
    plt.hist(data, bins=bins, edgecolor='black')
    plt.xlabel('Değer Aralığı')
    plt.ylabel('Frekans')
    plt.title('Veri Seti Histogramı')
    plt.show()

#KORELASYON ANALİZİ İÇİN
def plot_correlation_heatmap(data):
    sns.clustermap(data.corr(), annot=True, fmt=".2f")
    plt.show()

#OUTLİER DEĞİŞKENLERİ GRAFİK ÜZERİNDE GÖRMEK İÇİN
def plot_outliers(data):
    f, ax = plt.subplots(figsize=(20, 20))
    fig = sns.boxplot(data=data, orient="h")
    plt.show()


#MODELİN DEĞİŞKENLERİNİN ÖNEM SIRALAMASI
def plot_importance(model, features, num=None, save=False, name="importance.png"):
    if num is None:
        num = len(features.columns)

    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(name)


# BASE MODEL KURULUMU VE MODELLERİN SONUCLARI İÇİN
def base_models_results(X, y):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]
    results = {}  # Performans ölçütlerini saklamak için bir sözlük

    for name, classifier in classifiers:
        print(name)
        classifier_scores = {}
        for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            cv_results = cross_val_score(classifier, X, y, cv=5, scoring=score).mean()
            print(score + " score:" + str(cv_results))
            classifier_scores[score] = cv_results

        results[name] = classifier_scores
        print("\n")

    return results


#KATEGORİK KARDİNAL VE NÜMERİK KOLON AYIRMA
def grab_col_names(dataframe, cat_th=10,
                   car_th=20):  # essiz deger sayisi 10dan kucukse kategorik degisken, 5 den buyukse de kardinal degisken gibi dusunucez.
  # Veri setimiz küçük olduğundan ben 5 ile sınırlandırdım.
  cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

  num_but_cat = [col for col in dataframe.columns if
                 dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

  cat_but_car = [col for col in dataframe.columns if
                 dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

  cat_cols = num_but_cat + cat_cols
  cat_cols = [col for col in cat_cols if col not in cat_but_car]

  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
  num_cols = [col for col in num_cols if col not in cat_cols]

  print(f"Observations: {dataframe.shape[0]}")
  print(f"Variables: {dataframe.shape[1]}")
  print(f"Categorical Columns: {len(cat_cols)}")
  print(f"Numerical Columns: {len(num_cols)}")
  print(f"Categoric but Cardinal: {len(cat_but_car)}")
  print(f"Numeric but Categoric: {len(num_but_cat)}")

  return cat_cols, num_cols, cat_but_car

#numerik değiskenler ve target analizi
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


#nümerik değişken analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


#kategorik degisken analizi yani sadece outcome (Target)
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

#BOŞ DEĞERLERİ İŞARETLEME
def fill_zeros_with_nan(data, nan_col):
    data[nan_col] = data[nan_col].replace(0, np.NaN)
    return data

#BOŞ DEĞERLERİ DOLDURMA
def fill_missing_values(data):
    nan_col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data = fill_zeros_with_nan(data, nan_col)

    mean_value_outcome_1 = data.loc[data['Outcome'] == 1].mean()
    mean_value_outcome_0 = data.loc[data['Outcome'] == 0].mean()

    data.loc[data['Outcome'] == 1] = data.loc[data['Outcome'] == 1].fillna(mean_value_outcome_1)
    data.loc[data['Outcome'] == 0] = data.loc[data['Outcome'] == 0].fillna(mean_value_outcome_0)

    return data


#Aykırı değerimizi (Outlier) saptama işlemi için:
def outlier_thresholds(dataframe, col_name, q1 = 0.25, q3 = 0.75):
  quantile1 = dataframe[col_name].quantile(q1)
  quantile3 = dataframe[col_name].quantile(q3)
  interquantile_range = quantile3 - quantile1
  up_limit = quantile3 + 1.5 * interquantile_range
  low_limit = quantile1 - 1.5 * interquantile_range
  return low_limit, up_limit

#Thresholdlara göre outlier var mı yok mu diye kontrol etmek için:
def check_outlier(dataframe, col_name):
  low_limit, up_limit = outlier_thresholds(dataframe, col_name)
  if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
    return True
  else:
    return False

#Var olan outlierları görmek için:
def grab_outliers(dataframe, col_name, index=False):
  low, up = outlier_thresholds(dataframe, col_name)
  if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
    print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
  else:
    print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
  if index:
    outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
    return outlier_index

#replace_with_thresholds fonksiyonunu uygulamadan önce yukarıda kararını verdiğimiz 0 değerleri null yapmalıyız ki onları da outlier olarak görmesin.
def replace_with_thresholds(dataframe, variable):
  low_limit, up_limit = outlier_thresholds(dataframe, variable)
  dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
  dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

#FEATURE ENGINEERING
def data_process(df):
    # Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
    df.loc[(df['Age'] < 35), "NEW_AGE_CAT"] = 'young'
    df.loc[(df['Age'] >= 35) & (df['Age'] <= 55), "NEW_AGE_CAT"] = 'middleage'
    df.loc[(df['Age'] > 55), "NEW_AGE_CAT"] = 'old'

    # BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
    df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                             labels=["Underweight", "Healthy", "Overweight", "Obese"])

    # Glukoz degerini kategorik değişkene çevirme
    df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300],
                                 labels=["Normal", "Prediabetes", "Diabetes"])

    # BloodPressure
    df['NEW_BLOODPRESSURE'] = pd.cut(x=df['BloodPressure'], bins=[0, 79, 89, 123],
                                       labels=["normal", "hs1", "hs2"])

    # Insulin
    df['NEW_INSULIN'] = df['Insulin'].apply(lambda x: "Normal" if 16 <= x <= 166 else "Abnormal")
    cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=5, car_th=15)
    cat_cols = [col for col in cat_cols if "Outcome" not in col]
    data = one_hot_encoder(df, cat_cols, drop_first=True)
    cat_cols, num_cols, cat_but_car = grab_col_names(data,cat_th=5, car_th=15)
    replace_with_thresholds(data, "Insulin")
    X_scaled = StandardScaler().fit_transform(data[num_cols])
    X_scaled2 = StandardScaler().fit(data[num_cols])
    import joblib
    joblib.dump(X_scaled,"standart_scaler.pkl")
    y = data["Outcome"]
    X = data.drop(["Outcome"], axis=1)

    return X, y, data[num_cols]


def evaluate_model(model, X, y):
    # Modeli eğit ve tahminleri al
    predictions = model.predict(X)

    # Metrikleri hesapla
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    roc_auc = roc_auc_score(y, predictions)

    # Confusion matrix'i hesapla
    conf_matrix = confusion_matrix(y, predictions)

    return accuracy, f1, roc_auc, conf_matrix
