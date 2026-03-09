import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

LABEL_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

data_dir = "cifar-10-batches-py"

print("CIFAR-10 veri seti yükleniyor...")

X_train = []
y_train = []

for i in range(1,6):

    file = os.path.join(data_dir, "data_batch_" + str(i))

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    X_train.append(dict[b'data'])
    y_train += dict[b'labels']

X_train = np.concatenate(X_train)
y_train = np.array(y_train)

print("Training veri sayısı:", X_train.shape[0])

with open(os.path.join(data_dir, "test_batch"), 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

X_test = dict[b'data']
y_test = np.array(dict[b'labels'])

print("Test veri sayısı:", X_test.shape[0])


print("\nMesafe metriği seçin:")
print("1 -> L1 (Manhattan)")
print("2 -> L2 (Euclidean)")

distance_type = input("Seçim: ")

k = int(input("k değerini giriniz: "))

index = int(input("Test veri index giriniz (0-9999): "))

test_image = X_test[index]
true_label = y_test[index]

print("Gerçek sınıf:", true_label, "-", LABEL_NAMES[true_label])

distances = []

print("\nMesafeler hesaplanıyor...")

for i in range(len(X_train)):

    train_image = X_train[i]

    if distance_type == "1":
        distance = np.sum(np.abs(train_image - test_image))
    else:
        distance = np.sqrt(np.sum((train_image - test_image) ** 2))

    distances.append((distance, y_train[i]))


distances = sorted(distances, key=lambda x: x[0])

neighbors = distances[:k]

votes = {}

for d, label in neighbors:

    if label in votes:
        votes[label] += 1
    else:
        votes[label] = 1


predicted_class = max(votes, key=votes.get)

print("\nTahmin edilen sınıf:", predicted_class, "-", LABEL_NAMES[predicted_class])
print("Gerçek sınıf:", true_label, "-", LABEL_NAMES[true_label])

if predicted_class == true_label:
    print("Doğru tahmin")
else:
    print("Yanlış tahmin")


show = input("\nTest görüntüsünü göstermek ister misiniz? (e/h): ")

if show.lower() == "e":

    img = test_image.reshape(3, 32, 32)
    img = np.transpose(img, (1, 2, 0)) / 255.0

    plt.imshow(img)
    plt.title(f"Tahmin: {LABEL_NAMES[predicted_class]} | Gerçek: {LABEL_NAMES[true_label]}")
    plt.axis("off")
    plt.show()