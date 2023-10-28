import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore")


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

data = pd.read_csv('../OkulProje/database/diabetes.csv')

#Burada diabetes veri setini checkup ediyoruz.
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

check_df(data)
data.info()

# Kolon isimlerini büyültüyorum ki sorgulama vs yaparken yazmak, okumak kolay olsun.
data.columns = [col.upper() for col in data.columns]
data.head()

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


#Degiskenleri kategorik, kardinal ve kategorik olarak ayırıyoruz.
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
                   car_th=5):  # essiz deger sayisi 10dan kucukse kategorik degisken, 5 den buyukse de kardinal degisken gibi dusunucez.
  # Veri setimiz küçük olduğundan ben 5 ile sınırlandırdım.
  cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

  num_but_cat = [col for col in dataframe.columns if
                 dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

  cat_but_car = [col for col in dataframe.columns if
                 dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

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

#Boş kısımları ortalama ile doldurma işlemini bu kısımda değil, outlier değerleri baskıladıktan sonra yapma kararı aldım.
#0 değerlerini NaN olarak değiştirelim.
nan_col=['GLUCOSE','BLOODPRESSURE','SKINTHICKNESS', 'INSULIN', 'BMI']
data[nan_col]=data[nan_col].replace(0 ,np.NaN)
#kaç boş değer olduğunu gördük ki deri kalınlığı ve insülin değerinde fazlasıyla boş değer var.
data.isnull().sum()

#Aykırı değerimizi (Outlier) saptama işlemi için:
def outlier_thresholds(dataframe, col_name, q1 = 0.15, q3 = 0.85):
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


#Baskılama işlemi uyguladık şimdi tekrardan boş değerlerimizi kontrol edip, ortalama ile dolduralım.
data.isnull().sum()
"""
GLUCOSE            5
BLOODPRESSURE     35
SKINTHICKNESS    227
INSULIN          374
BMI               11
"""

#şimdi For döngüsü ile doldurulacak değişkenler içinde gezip,
# Outcome değeri 0 olan değerlerin ortalamasını Outcome değeri 0 olan boş değerlere;
# Outcome değeri 1 olan değerlerin ortalamasını Outcome değeri 1 olan boş değerlere atadım.
for i in nan_col:
    data[i][(data[i].isnull()) & (data["OUTCOME"] == 0)] = data[i][(data[i].isnull()) & (data["OUTCOME"] == 0)].fillna(data[i][data["OUTCOME"] == 0].mean())
    data[i][(data[i].isnull()) & (data["OUTCOME"] == 1)] = data[i][(data[i].isnull()) & (data["OUTCOME"] == 1)].fillna(data[i][data["OUTCOME"] == 1].mean())

data.isnull().sum()