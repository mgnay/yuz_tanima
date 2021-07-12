import face_recognition
import argparse
import pickle
import cv2
import os

TOLERANCE = 0.5

# dışarıdan görsel dizinini argüman olarak alma işlemi
parser = argparse.ArgumentParser()
parser.add_argument("gorsel_dizini", help="görselin dizini")
args = parser.parse_args()
dizin = args.gorsel_dizini

# pickle dosyasından eğitim verilerinin çekilmesi
data_encodings = "data_encodings.pickle"
detection_method: str = "cnn"  # veya hog

print("Yüz bilgileri yükleniyor...")
data = pickle.loads(open(data_encodings, "rb").read())
#  yükleme ve dlib formatına (RGB) dönüştürme
image = cv2.imread(dizin)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görseldeki yüzlerin konumunu bulma ve yüz bilgilerini arama
print("Yüzler tanımlanıyor...")
koordinat = face_recognition.face_locations(rgb, model=detection_method)
nitelikler = face_recognition.face_encodings(rgb, koordinat)
# bulunan her yüzün isimlerinin sırayla listelenmesi
bulunan_isimler = []

# loop over the facial embeddings
for nitelik in nitelikler:
    # attempt to match each face in the input image to our known
    # encodings
    bulunan_yuzler = face_recognition.compare_faces(data["encodings"], nitelik, TOLERANCE)
    isim = "Unknown"

    # check to see if we have found a match
    if True in bulunan_yuzler:
        index = [i for (i, b) in enumerate(bulunan_yuzler) if b]
        counts = {}
        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in index:
            isim = data["names"][i]
            counts[isim] = counts.get(isim, 0) + 1
        # determine the recognized face with the largest number of
        # votes (note: in the event of an unlikely tie Python will
        # select first entry in the dictionary)
        isim = max(counts, key=counts.get)

    # bulunan isimler
    bulunan_isimler.append(isim)

for ((top, right, bottom, left), isim) in zip(koordinat, bulunan_isimler):
    # Bulunan ismi yazdır
    cv2.rectangle(image, (left, top), (right, bottom), (20, 255, 20), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, isim, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (20, 255, 20), 2)


dosya_adi = os.path.splitext(dizin.split(os.path.sep)[-1])[0]
# çıktıyı göster
cv2.imwrite(dosya_adi+"_test.jpg", image)
print("Test görseli kaydedildi...")
cv2.waitKey(0)
print(bulunan_isimler)
