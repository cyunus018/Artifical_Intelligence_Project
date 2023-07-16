#Kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2. veri ön işleme
#2.1 veri yükleme
veriler = pd.read_csv('satislar.csv')
print(veriler)

aylar = veriler[['Aylar']]
print(aylar)

satislar =veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values
print(satislar2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test) 