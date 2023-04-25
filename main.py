from deepface import DeepFace
import json
import cv2
import matplotlib.pyplot as plt


def face_verify(img1, img2):
    try:
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)

        fig = plt.figure(figsize=(10, 7))
        rows = 1
        columns = 2

        fig.add_subplot(rows, columns, 1)
        plt.imshow(img1[:,:,::-1])
        plt.axis("off")
        plt.title("First")

        fig.add_subplot(rows, columns, 2)
        plt.imshow(img2[:, :, ::-1])
        plt.axis("off")
        plt.title("Second")

        plt.show()

        output = DeepFace.verify(img1_path=img1,
                                 img2_path=img2)

        if output.get("verified"):
            print("Эти фото одинаковы")
        else:
            print("Это разные фото")

        with open("res.json", "w") as file:
            json.dump(output, file, indent=4, ensure_ascii=False)

        return output
    except Exception:
        return "Не могу найти фото"


def face_find():
    try:
        result = DeepFace.find(img_path=r"faces\rayan1.jpeg",
                               db_path="Rayan")
        result = result.value.tolist()

        return result
    except Exception:
        return "Не могу найти фото"


def face_analyze():
    try:
        result = DeepFace.analyze(img_path=r"faces/skala1.jpg", actions=["age", "race", "gender", "emotions"])
        print(f'[+] Age: {result.get("age")}')
        print(f'[+] Gender: {result.get("gender")}')
        print(f'[+] Race:')
        for r, p in result.get("race").item():
            print(f"{r} - {round(p, 2)}%")
        print(f'[+] Emotions:')
        for e, p in result.get("emotions").item():
            print(f"{e} - {round(p, 2)}%")

    except Exception:
        return "Не могу найти фото"


def main():
    print(face_verify(img1="faces/ortega1.jpg", img2="faces/ortega2.jpg"))
    print(face_find())
    face_analyze()


if __name__ == "__main__":
    main()
