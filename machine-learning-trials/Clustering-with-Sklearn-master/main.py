from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cizdir(x,y,kumeler):
    sozluk={}
    renkler=['blue','red','green','gray','yellow','green']
    for i in range(len(kumeler)):
        try:
            sozluk[kumeler[i]].append([x[i],y[i]])
        except:
            sozluk[kumeler[i]]=[[x[i], y[i]]]
    
    adlar=[]
    for m in sozluk.keys():
        adlar.append(m)
#    adlar=sozluk.keys()

    for i in range(len(sozluk)):
        for j in range(len(sozluk[adlar[i]])):
            plt.scatter(sozluk[adlar[i]][j][0],sozluk[adlar[i]][j][1],c=renkler[i])

def ciz(yeni_veri,index1,index2,tahmin,gercek_degerler,yontem,merkezler=[]):
    cl=yeni_veri.columns.tolist()
    xl=cl[index1]
    yl=cl[index2]

    plt.subplot(1, 2, 1)
    plt.title("Gercek Veriler")
    plt.xlabel(xl)
    plt.ylabel(yl)
    cizdir(yeni_veri.iloc[:, index1].values, yeni_veri.iloc[:, index2].values, gercek_degerler)

    plt.subplot(1, 2, 2)
    plt.title(yontem+" Kumeleme")
    plt.xlabel(xl)
    plt.ylabel(yl)
    cizdir(yeni_veri.iloc[:, index1].values, yeni_veri.iloc[:, index2].values, tahmin)

    if len(merkezler) > 0:
        plt.scatter(merkezler[:,0],merkezler[:,1],c='black',label='Merkezler',s=30)
        plt.legend(loc='upper left')

    plt.show()

def bolunmeli(veri,k):
    kmeans = KMeans(n_clusters=k).fit(veri)
    #print kmeans.predict([[0.3,0.2,0.1,0.4,1]])
    return kmeans

def hiyerarsik(veri,k):
    clustering = AgglomerativeClustering(n_clusters=k).fit(veri)
    return clustering

def yogunluk_tabanli(veri,eps,min_samples):
    clustering = DBSCAN(eps=eps,min_samples=min_samples).fit(veri)
    return clustering

#MAIN
ad="Iris"
veri=pd.read_csv(ad+".csv")

sinif_sutun_indexi=5#int(raw_input("Sinif sutun indexini girin: "))#5#
kmeans_k=3#int(raw_input("Kmeans icin k yi girin: "))#3#
dbscan_eps=1#float(raw_input("DBSCAN icin eps yi girin: "))#0.45#
dbscan_min_samples=5#int(raw_input("DBSCAN icin min. ornek sayisini girin: "))#8#

veri=veri.drop('Id',axis=1)
sinif_sutun_indexi=4

#kategorik veri indexlerini bulma
sutun_adlari=veri.columns.tolist()
kategorik_sutun_adlari=[]
kategorik_sutun_indexleri=[]
for i in range(len(veri.columns.tolist())):
    if type(veri.iloc[1,i])==str:
        kategorik_sutun_adlari.append(sutun_adlari[i])
        kategorik_sutun_indexleri.append(i)

#kategorik verileri label encoder ile numerige cevirme
le=LabelEncoder()
yeni_veri=veri.copy()
for i in range(len(kategorik_sutun_adlari)):
    a=le.fit_transform(veri.iloc[:,kategorik_sutun_indexleri[i]])
    yeni_veri=yeni_veri.drop(kategorik_sutun_adlari[i],axis=1)#sikinti cikariyor
    s = {}
    s[kategorik_sutun_adlari[i]]=a
    yeni_veri=pd.concat([yeni_veri,pd.DataFrame(s)],axis=1)

#verinin sinif sutunu cikartiliyor
yeni_veri2=yeni_veri.copy()
yeni_veri=yeni_veri.drop(yeni_veri.columns.tolist()[sinif_sutun_indexi],axis=1)

#bolunmeli
kmeans=bolunmeli(yeni_veri.iloc[:,:].values,kmeans_k)
ciz(yeni_veri,0,1,kmeans.labels_,veri.iloc[:,sinif_sutun_indexi],"K-Means",kmeans.cluster_centers_)

#hiyerarsik
agg=hiyerarsik(yeni_veri,kmeans_k)
ciz(yeni_veri,2,3,agg.labels_,yeni_veri2.iloc[:,sinif_sutun_indexi],"Agglomerative")

#yogunluk
yog=yogunluk_tabanli(yeni_veri,dbscan_eps,dbscan_min_samples)
ciz(yeni_veri,0,3,yog.labels_,yeni_veri2.iloc[:,sinif_sutun_indexi],"DBScan")
