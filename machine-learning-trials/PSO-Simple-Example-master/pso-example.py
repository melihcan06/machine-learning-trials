# z=x^2+y^2 ; x,y E [0,10000] z maximization problem

import numpy as np
import math
import matplotlib.pyplot as plt

def parcaciklari_bas(parcaciklar):
    x=[]
    y=[]
    for i in parcaciklar:
        x.append(i.deger['x'])
        y.append(i.deger['y'])
    plt.scatter(x,y)
    plt.show()

MAX_DEGER=10000
def rastgele_sayi(alt=1,ust=10):
    return np.random.randint(alt,ust)

class Parcacik:
    def uygunluk_degeri_hesapla(self):#genel method
        #return self.deger['x'] ** 2 + self.deger['y'] ** 2
        t=0
        for i in self.deger:
            t+=self.deger[i]**2
        return t

    def __init__(self):
        self.deger = {'x': rastgele_sayi(), 'y': rastgele_sayi()}
        self.deger_uygunluk_degeri = 0#self.uygunluk_degeri_hesapla()#  # fitness
        self.pbest = {'x': 0, 'y': 0}
        self.pbest_uygunluk_degeri = 0 #self.deger_uygunluk_degeri# # fitness
        self.hiz = {'x': 0, 'y': 0}  # velocity

def pbestle_kendi_farki(parcacik):
    fark = {}
    for i in parcacik.deger:
        fark[i] = math.fabs(parcacik.pbest[i] - parcacik.deger[i])
    return fark

def gbestle_kendi_farki(parcacik,GBEST):
    fark = {}
    for i in parcacik.deger:
        fark[i] = math.fabs(GBEST.pbest[i] - parcacik.deger[i])
    return fark

def hiz_hesapla(parcacik,GBEST,c1=2, c2=2):  # velocity
    for i in parcacik.deger:
        parcacik.hiz[i] = parcacik.hiz[i] + c1 * rastgele_sayi(1,3) * (
        pbestle_kendi_farki(parcacik)[i]) + c2 * rastgele_sayi(1,3) * (gbestle_kendi_farki(parcacik,GBEST)[i])
    return parcacik

def hareket_ettir(parcacik,GBEST):  # hareket ettir,uyguluk hesapla ve pbesti kontrol et
    parcacik=hiz_hesapla(parcacik,GBEST)
    for i in parcacik.deger:
        parcacik.deger[i] = parcacik.deger[i] + parcacik.hiz[i]
        if parcacik.deger[i]>MAX_DEGER:
            parcacik.deger[i]=MAX_DEGER

    parcacik.uygunluk_degeri = parcacik.uygunluk_degeri_hesapla()

    if parcacik.deger_uygunluk_degeri > parcacik.pbest_uygunluk_degeri:  # pbest guncelleme
        parcacik.pbest = parcacik.deger.copy()
        parcacik.pbest_uygunluk_degeri = parcacik.deger_uygunluk_degeri

    return parcacik

def gbest_bul(parcaciklar):
    en_iyi_indis=0
    for i in range(1,len(parcaciklar)):
        if parcaciklar[i].deger_uygunluk_degeri>parcaciklar[en_iyi_indis].deger_uygunluk_degeri:
            en_iyi_indis=i

    return parcaciklar[en_iyi_indis]

def baslangic_parcaciklari_olustur(baslangic_parcacik_sayisi=5):
    parcaciklar=[]
    for i in range(baslangic_parcacik_sayisi):
        parcaciklar.append(Parcacik())
    return parcaciklar

dongu_sayisi=5
parcaciklar=baslangic_parcaciklari_olustur(2)
GBEST=gbest_bul(parcaciklar)

for i in range(dongu_sayisi):
    for j in parcaciklar:
        #j.deger_uygunluk_degeri=j.uygunluk_degeri_hesapla()
        j=hareket_ettir(j,GBEST)

    for j in parcaciklar:# gbest guncelleme
        if j.deger_uygunluk_degeri > GBEST.pbest_uygunluk_degeri:
            GBEST.pbest = j.deger.copy()
            GBEST.pbest_uygunluk_degeri = j.deger_uygunluk_degeri
    
    parcaciklari_bas(parcaciklar)
