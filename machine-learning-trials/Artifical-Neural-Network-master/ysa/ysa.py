import numpy as np

class ysa:
    def __init__(self, giris, cikis, ok, akns):
        self.giris = giris
        self.cikis = cikis  # beklenen
        self.ok = ok  # ogrenme katsayisi
        self.akns = akns  # ara katman noron sayisi
        self.agirliklar1 = np.random.random((giris.shape[1], akns))#np.array([[0.2, 0.7], [-0.1, -1.2], [0.4, 1.2]])#
        self.agirliklar2 = np.random.random((akns, cikis.shape[1]))#np.array([[1.1, 3.1], [0.1, 1.17]])#
        self.dongu_hatasi=0
        self.toplam_hata=0

    def sigmoid(self, x):
        return 1 / (1 + np.e ** -x)

    def sigmoid_turev(self, x):
        return x * (1 - x)

    def ileri_yayilim(self, girdi=None):
        try:
            if girdi == None:
                girdi = self.giris
        except:
            girdi=girdi

        self.ara_katman1 = self.sigmoid(np.dot(girdi, self.agirliklar1))
        self.cikti = self.sigmoid(np.dot(self.ara_katman1, self.agirliklar2))
        return self.cikti

    def egitim(self, dongu=1):
        i = 0

        while (i < dongu):

            self.ileri_yayilim()
            self.hata = self.cikis - self.cikti  # baska yontemde secilebilir

            self.cikti_sigmoid_hatasi = self.hata * self.sigmoid_turev(self.cikti)
            self.Err = np.dot(self.cikti_sigmoid_hatasi, self.agirliklar2.T)
            self.ara_katman_sigmoid_hatasi = self.sigmoid_turev(self.ara_katman1) * (self.Err)

            self.agirliklar1 += np.dot(self.giris.T * self.ok, self.ara_katman_sigmoid_hatasi)
            self.agirliklar2 += np.dot(self.ara_katman1.T * self.ok, self.cikti_sigmoid_hatasi)

            xx=(np.sum(np.multiply(self.hata,self.hata)))/2
            print("Dongu: " + str(i) + "  Dongu hatasi: " + str(xx))
            self.toplam_hata += xx
            i += 1

        print("Toplam hata: " + str(self.toplam_hata))


import pandas as pd
veri=pd.read_csv("veri.csv")

sinif_kolon_sayisi=2
x_boyutu=veri.shape[0],veri.shape[1]-sinif_kolon_sayisi
y_boyutu=veri.shape[0],sinif_kolon_sayisi

x=np.array(veri.iloc[:,0:-sinif_kolon_sayisi].values).reshape(x_boyutu)
y=np.array(veri.iloc[:,-sinif_kolon_sayisi:].values).reshape(y_boyutu)

noron_sayisi=int(input("Noron sayisini girin:"))#5#
a = ysa(x,y, 0.1, noron_sayisi)

a.egitim(100)

print(a.ileri_yayilim(np.array(veri.iloc[3,0:-sinif_kolon_sayisi].values).reshape(1,veri.shape[1]-sinif_kolon_sayisi)))
