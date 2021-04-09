import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def outlier_indexs(df, q1, q3):
    outliers = []
    iqr = q3 - q1
    alt = q1 - 1.5 * iqr
    ust = q3 + 1.5 * iqr
    j = 0

    for i in df.values:
        if i > ust or i < alt:
            outliers.append(j)
        j += 1

    return outliers

def ozellikler(ham_veri,class_index):
    veri = ham_veri.copy()

    print("TITLE: " + dosya_adi + "\n")

    # kolon adlari
    print("COLUMN NAMES:\n\n")
    print(str(ham_veri.columns.tolist()) + "\n")

    # numerik kolonlari bulma
    numerik_kolon_indexleri = []  # print(veri.get_dtype_counts())

    j = 0
    for i in ham_veri.iloc[1, :]:
        if type(i) != str:
            numerik_kolon_indexleri.append(j)
        j += 1

    print(str(len(numerik_kolon_indexleri)) + " VALUES SUMMERIZATION\n")

    # ozellik bastirma
    for i in numerik_kolon_indexleri:
        veri = veri.sort_values([veri.columns[i]], axis=0)
        median = veri.iloc[:, i].median()
        median_index = int(len(veri.iloc[:, i]) / 2)
        q1 = veri.iloc[:, i].quantile(0.25)
        q3 = veri.iloc[:, i].quantile(0.75)

        # iqr = q3 - q1
        # alt = q1 - 1.5 * iqr
        # ust = q3 + 1.5 * iqr

        print(str(veri.columns[i]) + "\n" + \
              "Min:" + str(veri.iloc[:, i].min()) + "\n" +
              "Max:" + str(veri.iloc[:, i].max())) + "\n" + \
             "Median:" + str(median) + "\n" + \
             "Q1:" + str(q1) + "\n" + \
             "Q3:" + str(q3) + "\n"  # +\
        # "Sapan veri siniri:"+str(float(int(alt*100))/100)+"      "+str(float(int(ust*100))/100)

        oi = outlier_indexs(ham_veri.iloc[:, i], q1, q3)

        print "OUTLIER INDEXS"

        if len(oi) == 0:

            print "Outlier sample not found"

        else:

            for j in oi:
                print j

        print("-----------------------------------")

    class_name = ham_veri.columns[class_index]

    print "CLASS LABELS AND COUNTS"
    print ham_veri.groupby(class_name)[class_name].count()

def min_max_norm(sutun, adi):
    ek = min(sutun)
    eb = max(sutun)
    yeni_degerler = []
    yeni_dic = {}
    for i in sutun:
        yeni_degerler.append((i - ek) / float(eb - ek))

    yeni_dic[adi] = yeni_degerler

    return pd.DataFrame(yeni_dic)

def normalizasyon(veri, sutun_indexi):
    sutun_adlari = veri.columns
    veri = veri.drop(sutun_adlari[sutun_indexi], axis=1)
    yeni_sutunlar = []

    for i in range(len(sutun_adlari) - 1):
        yeni_sutunlar.append(min_max_norm(veri.iloc[:, i], sutun_adlari[i]))

    yeni_frame = yeni_sutunlar[0]
    for i in range(1, len(yeni_sutunlar)):
        yeni_frame = pd.concat([yeni_frame, yeni_sutunlar[i]], axis=1)

    return yeni_frame

#KNN
def knn(yeni_min_uygulanmis,k,sinif_belirleme,sutun_indexi,veri_norm,veri):
    yeni_uzakliklar = []

    # oklid uygulaniyor
    for i in range(veri_norm.shape[0]):
        t = 0.0
        for j in range(veri_norm.shape[1]):
            t += (yeni_min_uygulanmis.iloc[j] - veri_norm.iloc[i, j]) ** 2
        yeni_uzakliklar.append(sqrt(t))

    min_uzakliklar = []  # [degeri,indexi]
    indexler = []
    yeni_uzakliklar_yedek = yeni_uzakliklar[:]

    # en kisa k tane uzaklik bulunuyor
    for i in range(k):
        a = min(yeni_uzakliklar)
        b = yeni_uzakliklar_yedek.index(a)
        indexler.append(b)
        c = yeni_uzakliklar_yedek[indexler[i]]
        yeni_uzakliklar.remove(c)

    # k tane kisanin nitelikleri aliniyor
    nitelikler = []
    for i in indexler:
        nitelikler.append(veri.iloc[i, sutun_indexi])

    # sinif sutununun nitelikleri
    nitelik_adlari = veri.groupby(veri.columns[sutun_indexi])[veri.columns[sutun_indexi]].count().keys().tolist()
    sayilar = np.zeros((len(nitelik_adlari, )))

    # sinif belirleme

    # en cok tekrarlanan
    if sinif_belirleme == 1:
        # hangi nitelikten kac tane var sayiliyor
        for i in range(len(nitelikler)):
            for j in range(len(nitelik_adlari)):
                if nitelikler[i] == nitelik_adlari[j]:
                    sayilar[j] += 1
                    break

        # en cok tekrarlanan
        for i in range(len(sayilar)):
            if sayilar[i] == np.max(sayilar):
                # print "\n" + str(girdi[:])
                # print str(nitelik_adlari[i]) + " sinifina aittir."
                return nitelik_adlari[i]
                # break

    # agirlikli
    else:
        # agirliklar hesaplaniyor
        for i in range(k):
            for j in range(len(nitelik_adlari)):
                if nitelikler[i] == nitelik_adlari[j]:
                    a = round(yeni_uzakliklar_yedek[indexler[i]], 2)
                    b = a * a
                    c = round(1 / (b), 2)
                    sayilar[j] += c
                    break

        # en buyuk bulunuyor
        eb = 0 #eb indextir
        for i in range(1,len(sayilar)):
            if sayilar[i] > eb:
                eb = i

        # print "\n" + str(girdi[:])
        # print str(nitelik_adlari[eb]) + " sinifina aittir."
        return nitelik_adlari[eb]


#MAIN

dosya_adi = "Iris"  
veri = pd.read_csv(dosya_adi + ".csv")

#Veri setini karistiriyorum
from sklearn.utils import shuffle
veri = shuffle(veri)
# girdiler
oto=int(raw_input("girdiler otomatik ayarlansin mi?(0=hayir,1=evet): "))

if oto == 0:
    sutun_indexi = int(raw_input("sutun indexini giriniz: "))
    k = int(raw_input("k yi giriniz: "))
    sinif_belirleme = int(raw_input("sinif belirleme turunu giriniz(en kucuk uzaklik = 0 - agirlikli = 1): "))
    girdi = raw_input("Degerleri giriniz(ornek = 0, 4.9, 3, 1.4, 0.2):")
    girdi = girdi.split(',')
    girdi = [float(i) for i in girdi]

else:
    if dosya_adi == "Iris":
        sutun_indexi =  5
        k =  4
        sinif_belirleme =  int(raw_input("sinif belirleme turunu giriniz(en kucuk uzaklik = 0 - agirlikli = 1): "))
        girdi = '0, 4.9, 3, 1.4, 0.2'
        girdi = girdi.split(',')
        girdi = [float(i) for i in girdi]

ozellikler(veri,sutun_indexi)


# veri on isleme
sutunlar = veri.columns.tolist()
sutunlar.remove(sutunlar[sutun_indexi])

a = {}
m = 0
for i in sutunlar:
    a[i] = [girdi[m]]
    m += 1

a[veri.columns.tolist()[sutun_indexi]] = "Nan"
a = pd.DataFrame(a)

sahte_veri = veri.append(a, sort=False)#veri + yeni gelecek
veri_norm = normalizasyon(sahte_veri, sutun_indexi)
girdi_norm = veri_norm.iloc[-1, :]
veri_norm = veri_norm.iloc[:-1, :]

#Test Train ayriliyor
test_sayisi = int(veri_norm.shape[0]*0.2)
veri_test = veri_norm.iloc[veri_norm.shape[0]-test_sayisi:veri_norm.shape[0],:]
veri_norm = veri_norm.iloc[:veri_norm.shape[0]-test_sayisi,:]

tahminler = []
for i in range(veri_test.shape[0]):#len(test)
    tahminler.append(knn(veri_test.iloc[i,:],k,sinif_belirleme,sutun_indexi,veri_norm,veri))

tahmin_skor ={}
tahmin_skor['dogru']=0
tahmin_skor['yanlis']=0

for i in range(len(tahminler)):
    b=veri.iloc[veri.shape[0]-test_sayisi+i,sutun_indexi]
    c=tahminler[i]
    if  c == b:
        tahmin_skor['dogru']+=1
    else:
        tahmin_skor['yanlis']+=1

print "\n\n"+"KNN"
print "Test verisi uzerindeki deneme: "+ str(tahmin_skor)
print "Girdimiz: "+str(girdi)
print "Girdimizin sonucu : "+str(knn(girdi_norm,k,sinif_belirleme,sutun_indexi,veri_norm,veri))
