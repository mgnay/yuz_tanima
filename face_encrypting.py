from imutils import paths
import face_recognition
import pickle
import cv2
import os

EGITIM_DIZINI = "../input/hababam-snf/training"

print("Resimler alınıyor...")
resimDizinleri = list(paths.list_images(EGITIM_DIZINI))

encodingData = []
isimDizisi = []

for (i, dizin) in enumerate(resimDizinleri):
    # Dizin adından kişi ismini bul
    print("Resimler  {}/{}".format(i + 1, len(resimDizinleri)))
    isim = dizin.split(os.path.sep)[-2]
    
    # BGR (opencv) formatındaki resimleri RGB(dlib) formatına dönüştür
    image = cv2.imread(dizin)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    while True:
        try:
            yuzKonumu = face_recognition.face_locations(rgb, model="cnn")
            break
        except:
            rgb = cv2.resize(rgb, (int(rgb.shape[0]/2), int(rgb.shape[1]/2)))

    # Yüz niteliklerinin hesaplanması
    encodings = face_recognition.face_encodings(rgb, yuzKonumu)

    # loop over the encodings
    for data in encodings:
        encodingData.append(data)
        isimDizisi.append(isim)

print("Yüz verileri kaydediliyor...")
data = {"encodings": encodingData, "names": isimDizisi}
file = open("data_encodings.pickle", "wb")
file.write(pickle.dumps(data))
file.close()
