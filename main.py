#---СКРИПТ ДЛЯ ПОДСЧЕТА КОЛИЧЕСТВА ОБЬЕКТОВ НА ВИДЕО.

import cv2
import pafy
import rgb as rgb
import tensorflow as tf
import tensorflow as hub
from tensorflow import keras

timeLaps = 10       #---Указывается промежуток времени для создания скриншотов.
#---Переменная - ссылка на видео поток. Если используем видеофайл, пишем его имя.
#---Если это камера видеонаблюдения, пишем ссылку на этот поток.
#---Например: (http:// - указываем протокол, затем ip камеры 192.168.1.88,
#---После ip камеры обычно задается порт к которому нужно подключаться ":443".
#---И ссылка на сам видеопоток "/media/video".
#---Общий вид - "http://192.168.1.88:443/media/video"
#url = "new_box.mp4.avi"
url = "https://www.youtube.com/watch?v=rG13FY2ytno"

video = pafy.new(url)                                       #---Создаем объект видео, если видео из YouTube.
best = video.getbest(preftype="mp4")                        #---Выбираем лучшее качество в формате mp4.

capture = cv2.VideoCapture(best.url)                        #---Создается объект камеры в который передается ссылка на видеопоток.


count = 0                                                   #---Счетчик кадров.
fps = int(capture.get(cv2.CAP_PROP_FPS))  #_fps камеры.
success = True                                              #---Удачное чтение кадра
i = 0                                                       #---Переменная для нумерации скриншота.

detector = keras.models.load_model("faster_rcnn_openimages_v4_inception_resnet_v2_1").signatures["default"]  #---Создаем нейросеть. Загружается с помощью метода load, класса hub. В скобках указывается путь к модели. Для этой модели загружаем сигнатуру по умолчанию.
                                                                                                             #---После получения объекта нейросети, мы можем ее использовать передавая в нее изображение success, img = capture.read().




#---Создаем цикл чтения кадров с видеопотока.
while success:
    success, img = capture.read()                                                      #---Прочитали кадр из видеопотока и поместили изображение в переменную img. При вызове метода read мы читаем только один текущий кадр.
    if count % (timeLaps*fps) == 0:                                                    #---Чтобы сохранять каждый 10 кадр.
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                     #---Так как нейросеть чувствительна к цвету - преобразовываем.
        converted_img = tf.image.convert_image_dtype(rgb, tf.float32)[tf.newaxis, ...] #---Полученное изображение конветрируем в tensorflow для передачи в detector.
        result = detector(converted_img)                                               #---Запускаем нейросеть, передав в нее наше сконвертированное изображение.
        result = {key: value.numpy() for key, value in result.items()}                 #---Получаем в результате tensor, и для удобной работы преобразовываем его в словарь.
        count_object = 0                                                               #---Счетчик объектов.
        h, w, c = rgb.shape                                                            #---Для определения ширины и высоты изображения (для отладки)
        # ---Теперь нужно сделать фильтр для выделения только нужных нам объектов..
        for j in range(len(result["detection_class_entities"])):
            if result["detection_class_entities"][j] in (b"Person", b"Man", b"Woman") and result["detection_scores"][j] > 0.1:
                count_object += 1
                #---Для отладки визуализируем то, что скрипт будет находить.
                box = result["detection_boxes"][j]
                cv2.rectangle(rgb, (int(box[1] * w), int(box[0] * h)),
                              (int(box[3] * w), int(box[2] * h), (0, 255, 0), 2))      #---Нарисуем прямоугольник-рамку.
        cv2.imwrite("scr" + str(i) + ".png", img)
        i += 1                                                                         #---При каждом сохранении скриншота, счетчик скриншота увеличиваем.
        print(count_object)
    count+=1                                                                           #---И при каждом чтении кадра, увеличиваем счетчик кадров.

capture.release()                                                                      #---В конце очищаем память.
cv2.destroyAllWindows()







