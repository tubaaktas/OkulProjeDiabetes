import joblib
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scikitplot.metrics import plot_roc_curve
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

data = pd.read_csv("database/diabetes.csv")

# Kolon isimlerini büyültüyorum ki sorgulama vs yaparken yazmak, okumak kolay olsun.
data.columns = [col.upper() for col in data.columns]
data.head()
"""
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

"""
#check_df(data)
"""
#Outlier değerleri grafik üzerinde görebilmek için
f, ax = plt.subplots(figsize=(20,20)) #f->figure and ax->axis
fig = sns.boxplot(data=data, orient="h") #horizontally (grafiği yatayda alabilmek için)
#plt.show()


#Korelasyon analizi için
#Mesela buradaki korelasyona bakarak, doğum ve yaş arasında 0.54 pozitif korelasyon var.
#Outcome ı en çok etkileyen glikoz değeriymiş. 0.47
#Glikoz değerinden sonra en çok etki eden BMI olmuş. 0.29
#Yaş ile deri kalınlığı arasında da negatif korelasyon bulunmakta. -0.11
sns.clustermap(data.corr(), annot = True, fmt = ".2f")
#plt.show()
"""
#CONFUSION MATRİX İÇİN
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
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

# BASE MODEL KURULUMU
def base_models(X, y):
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

y = data["OUTCOME"]
X = data.drop("OUTCOME", axis=1)

model_results = base_models(X,y)
# Her bir performans metriği için grafik oluşturma
metrics = list(model_results[list(model_results.keys())[0]].keys())

for metric in metrics:
    plot_metric(metric, model_results, title="base model için")


def plot_importance(model, features, num=len(X), save=True):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    #plt.show()
    if save:
        plt.savefig('importances.png')

#Degiskenleri kategorik, kardinal ve nümerik olarak ayırıyoruz.
# Kardinal Degisken : kategorik degiskenin 20den fazla (buna biz karar veriyoruz 5 da  yazabilirdik.) sınıfı varsa kategorik gibi gorunen degiskenlerdir.
"""
Kardinal değişkenler, sayılarla ifade edilebilen ve belirli bir sıralamaya sahip olmayan değişkenlerdir. 
Örneğin, bir ankette katılımcıların yaşları, eğitim seviyeleri veya gelir düzeyleri kardinal değişkenlere örnek olarak verilebilir.
Aşağıda, katılımcıların yaşlarını içeren basit bir veri seti örneği bulunmaktadır:

{25,30,35,40,22,28,32,37,45,29}

Bu veri setindeki her bir sayı, bir katılımcının yaşı olarak temsil edilir. 
Bu veri seti, kardinal bir değişkeni gösterir çünkü her bir değer sayısal olarak ifade edilebilir ve bu değerler arasında bir sıralama mevcuttur.
"""
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

cat_cols, num_cols, cat_but_car = grab_col_names(data)

"""
Observations: 768
Variables: 9
Categorical Columns: 1
Numerical Columns: 8
Categoric but Cardinal: 0
Numeric but Categoric: 1
(['OUTCOME'], ['PREGNANCIES', 'GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI', 'DIABETESPEDIGREEFUNCTION', 'AGE'], [])
#Çıktıda aldığımız değerlere göre kardinal bir değişkenimiz bulunmamakta, 1 kategorik, 8 numerik, 1 tane de numerik ama kategorik (OUTCOME) değerimiz bulunmakta.
"""

#numerik değiskenler ve target analizi
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(data, "OUTCOME", col)

#nümerik değişken analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(data, col, plot=False)

#kategorik degisken analizi yani sadece outcome
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(data, "OUTCOME")

#0 değerlerini NaN olarak değiştirelim.
nan_col=['GLUCOSE','BLOODPRESSURE','SKINTHICKNESS', 'INSULIN', 'BMI']
data[nan_col]=data[nan_col].replace(0, np.NaN)
#kaç boş değer olduğunu gördük ki deri kalınlığı ve insülin değerinde fazlasıyla boş değer var.
data.isnull().sum()


# "Outcome" değeri 1 olan gözlemler için medyan değer hesaplama
median_value_outcome_1 = data.loc[data['OUTCOME'] == 1].median()
# "Outcome" değeri 0 olan gözlemler için medyan değer hesaplama
median_value_outcome_0 = data.loc[data['OUTCOME'] == 0].median()
# Boş değerleri doldurma
data.loc[data['OUTCOME'] == 1] = data.loc[data['OUTCOME'] == 1].fillna(median_value_outcome_1)
data.loc[data['OUTCOME'] == 0] = data.loc[data['OUTCOME'] == 0].fillna(median_value_outcome_0)

data.isnull().sum()

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

for col in num_cols:
  print(col,"için LOW LIMIT, UP LIMIT değerleri = ", outlier_thresholds(data,col),"\n")

for col in num_cols:
  print(col,"için outlier kontrolü", check_outlier(data,col),"\n")


#Bu çıktıdan alınan değerlere göre hamilelik (PREGNANCIES) 14-17 aralığındakiler outlier,
#Glikoz değerleri için 0 görünen kısımlar null (glikoz 0 olamaz),
#Kan basıncı değeri (BLOODPRESSURE) 0 olamaz null değerler var.
#BMI değerleri 0 olamaz null değerler var.
#Deri kalınlığı min 0.6 mm, max 2.4mm dir. 0 olan değerler null değerdir.
for col in num_cols:
  print(col,"için var olan outlierlar\n", grab_outliers(data,col))

#replace_with_thresholds fonksiyonunu uygulamadan önce yukarıda kararını verdiğimiz 0 değerleri null yapmalıyız ki onları da outlier olarak görmesin.
def replace_with_thresholds(dataframe, variable):
  low_limit, up_limit = outlier_thresholds(dataframe, variable)
  dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
  dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#şimdi kalan aykırı değerlerin değişimini sağlayabiliriz.
for col in num_cols:
  print(replace_with_thresholds(data, col))

"""
plt.hist(data, bins=10, edgecolor='black')  # 'bins' parametresiyle aralık sayısını belirleyebilirsiniz
plt.xlabel('Değer Aralığı')
plt.ylabel('Frekans')
plt.title('Veri Seti Histogramı')
plt.show()
"""
#data["AGE"].min()

#FEATURE ENGINEERING

# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
data.loc[(data['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
data.loc[(data['AGE'] >= 35) & (data['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
data.loc[(data['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
data['NEW_BMI'] = pd.cut(x=data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                         labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Glukoz degerini kategorik değişkene çevirme
data["NEW_GLUCOSE"] = pd.cut(x=data["GLUCOSE"], bins=[0, 140, 200, 300],
                             labels=["Normal", "Prediabetes", "Diabetes"])

# BloodPressure
data['NEW_BLOODPRESSURE'] = pd.cut(x=data['BLOODPRESSURE'], bins=[0, 79, 89, 123],
                                   labels=["normal", "hs1", "hs2"])

# Insulin
data['NEW_INSULIN'] = data['INSULIN'].apply(lambda x: "Normal" if 16 <= x <= 166 else "Abnormal")

data["INSULIN"].min()
data["INSULIN"].max()

data.head()

'''
# Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma
data.loc[(data["BMI"] < 18.5) & ((data['AGE'] >= 35) & (data['AGE'] <= 55)), "NEW_AGE_BMI_NOM"] = "underweight_middleage"

# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
data.loc[(data["GLUCOSE"] < 70) & ((data['AGE'] >= 35) & (data['AGE'] <= 55)), "NEW_AGE_GLUCOSE_NOM"] = "low_middleage"
'''
"""

# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
data.loc[(data["AGE"] >= 18) & (data["AGE"] <= 32), "NEW_AGE_CAT"] = "Young"
data.loc[(data["AGE"] >  32) & (data["AGE"] <  50), "NEW_AGE_CAT"] = "Adult"
data.loc[(data["AGE"] >= 50), "NEW_AGE_CAT"] = "Mature"

# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
data['NEW_BMI'] = pd.cut(x=data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Glukoz degerini kategorik değişkene çevirme
data["NEW_GLUCOSE"] = pd.cut(x=data["GLUCOSE"], bins=[0, 70, 140, 200, 300], labels=["Low", "Healthy", "Prediabetes", "Diabetes"])

# Deri Kalinligi degerini kategorik degiskene cevirme mm turunden
data["NEW_SKIN_THIC"] = pd.cut(x=data["SKINTHICKNESS"], bins=[1.25, 2, 2.5, 3.25], labels=["Thin", "Healthy", "Thick"])

# # Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma 3 kırılım yakalandı
data.loc[(data["BMI"] < 18.5) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "UnderweightYoung"
data.loc[(data["BMI"] < 18.5) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "UnderweightAdult"
data.loc[(data["BMI"] < 18.5) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "UnderweightMature"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "HealthyYoung"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "HealthyAdult"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "HealthyMature"
data.loc[((data["BMI"] >= 25) & (data["BMI"] < 30)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "OverweightYoung"
data.loc[((data["BMI"] >= 25) & (data["BMI"] < 30)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "OverweightAdult"
data.loc[((data["BMI"] >= 25) & (data["BMI"] < 30)) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "OverweightMature"
data.loc[(data["BMI"] > 30) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "ObeseYoung"
data.loc[(data["BMI"] > 30) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "ObeseAdult"
data.loc[(data["BMI"] > 30) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "ObeseMature"

# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
data.loc[(data["GLUCOSE"] < 70) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "LowYoung"
data.loc[(data["GLUCOSE"] < 70) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "LowAdult"
data.loc[(data["GLUCOSE"] < 70) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "LowMature"
data.loc[((data["GLUCOSE"] >= 70) & (data["GLUCOSE"] < 140)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "HealthyYoung"
data.loc[((data["GLUCOSE"] >= 70) & (data["GLUCOSE"] < 140)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "HealthyAdult"
data.loc[((data["GLUCOSE"] >= 70) & (data["GLUCOSE"] < 140)) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "HealthyMature"
data.loc[((data["GLUCOSE"] >= 140) & (data["GLUCOSE"] <= 200)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesYoung"
data.loc[((data["GLUCOSE"] >= 140) & (data["GLUCOSE"] <= 200)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesAdult"
data.loc[((data["GLUCOSE"] >= 140) & (data["GLUCOSE"] <= 200)) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesMature"
data.loc[(data["GLUCOSE"] > 200) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesYoung"
data.loc[(data["GLUCOSE"] > 200) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesAdult"
data.loc[(data["GLUCOSE"] > 200) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesMature"

# Deri Kalinligi ve Beden Kitle Indeksi degerlerini bir arada dusunerek kategorik degisken olusturma
data.loc[(data["BMI"] < 18.5) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "UnderweightThin"
data.loc[(data["BMI"] < 18.5) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "UnderweightHealthy"
data.loc[(data["BMI"] < 18.5) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "UnderweightThick"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyThin"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "Healthy_Healthy"
#data.loc[((data["BMI"] >= 18.5) % (data["BMI"] < 25)) & ((data["SKINTHICKNESS"] > 2 & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "Healthy_Healthy"
data.loc[((data["BMI"] >= 18.5)) & (data["BMI"] < 25) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyThick"
data.loc[((data["BMI"] >= 25) % (data["BMI"] < 30)) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "OverweightThin"
data.loc[((data["BMI"] >= 25) % (data["BMI"] < 30)) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "OverweightHealthy"
data.loc[((data["BMI"] >= 25) % (data["BMI"] < 30)) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "OverweightThick"
data.loc[(data["BMI"] > 30) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "ObeseThin"
data.loc[(data["BMI"] > 30) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "ObeseHealthy"
data.loc[(data["BMI"] > 30) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "ObeseThick"

# Deri Kalinligi ve Insulin degerlerini bir arada dusunerek kategorik degisken olusturma
data.loc[((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)) & ((data["INSULIN"] >= 16) & (data["INSULIN"] <=160)), "NEW_BMI_SKIN_THIC_NOM"] = "ThinNormal"
data.loc[((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)) & (data["INSULIN"] >=160), "NEW_BMI_SKIN_THIC_NOM"] = "ThinAbNormal"
data.loc[((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)) & ((data["INSULIN"] >= 16) & (data["INSULIN"] <=160)), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyNormal"
data.loc[((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)) & (data["INSULIN"] >=160), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyAbNormal"
data.loc[((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3)) & ((data["INSULIN"] >= 16) & (data["INSULIN"] <=160)), "NEW_BMI_SKIN_THIC_NOM"] = "ThickNormal"
data.loc[((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3)) & (data["INSULIN"] >=160), "NEW_BMI_SKIN_THIC_NOM"] = "ThickAbNormal"

# Insulin
data['NEW_INSULIN'] = data['INSULIN'].apply(lambda x: "Normal" if 16 <= x <= 166 else "Abnormal")
"""

cat_cols, num_cols, cat_but_car=grab_col_names(data,cat_th=5, car_th=15)
cat_cols = [col for col in cat_cols if "OUTCOME" not in col]
num_cols

#One Hot Encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

data = one_hot_encoder(data,cat_cols, drop_first=True)

data.columns = [col.upper() for col in data.columns]
cat_cols, num_cols, cat_but_car = grab_col_names(data, cat_th=5, car_th=15)

#Standart Scaler
X_scaled = StandardScaler().fit_transform(data[num_cols])
data[num_cols] = pd.DataFrame(X_scaled, columns=data[num_cols].columns)

#data.head()
y = data["OUTCOME"]
X = data.drop(["OUTCOME"], axis=1)
#check_df(data)

model_results = base_models(X, y)

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        plot_roc_curve(model, X_test, y_test, name=name)

    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend()
    plt.show()

# Her bir performans metriği için grafik oluşturma
metrics = list(model_results[list(model_results.keys())[0]].keys())

for metric in metrics:
    plot_metric(metric, model_results, title="feature engineering sonrası")

#Random Forest Model
rf_model = RandomForestClassifier(random_state=46)

rf_params = {
    "n_estimators": [75, 150, 250],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ['auto', 'sqrt']
}

# GridSearchCV kullanarak en iyi parametreleri bulmak için
rf_gs_best = GridSearchCV(rf_model,
                          rf_params,
                          cv=3,
                          n_jobs=-1,
                          verbose=True).fit(X, y)


# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# En iyi parametrelerle final modeli oluşturmak için
final_rf_model = rf_model.set_params(**rf_gs_best.best_params_).fit(X, y)

# Tahmin yapma
y_pred = rf_model.predict(X_test)

# Confusion matrix oluşturma
conf_matrix = confusion_matrix(y_test, y_pred)
# Heatmap ile confusion matrix görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plot_importance(final_rf_model, X, save=True)




