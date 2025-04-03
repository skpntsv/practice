import time
import sys
import numpy as np
import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import gzip
import onnx
import tf2onnx
import onnxruntime as ort # Для проверки ONNX
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# --- Класс загрузки данных (ваш класс) ---
class MNISTDataLoader:
    def __init__(self, data_dir, one_hot=True, class_num=10):
        self.data_dir = data_dir
        self.one_hot = one_hot
        self.class_num = class_num

        # Проверяем существование директории
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Директория данных не найдена: {self.data_dir}")

        print(f"Загрузка данных из: {self.data_dir}")
        self.train_images, self.train_labels = self._load_data('train')
        self.test_images, self.test_labels = self._load_data('t10k')
        print("Данные загружены.")
        print(f"  Train: {self.train_images.shape}, {self.train_labels.shape}")
        print(f"  Test:  {self.test_images.shape}, {self.test_labels.shape}")

        # Добавляем свойство для количества тренировочных примеров
        self.num_train_examples = len(self.train_images)

    def _load_data(self, prefix):
        """Load MNIST data from .gz files"""
        image_path = os.path.join(self.data_dir, f'{prefix}-images-idx3-ubyte.gz')
        label_path = os.path.join(self.data_dir, f'{prefix}-labels-idx1-ubyte.gz')

        if not os.path.exists(image_path) or not os.path.exists(label_path):
             raise FileNotFoundError(f"Файлы данных MNIST не найдены в {self.data_dir} с префиксом '{prefix}'")

        try:
            # Load images
            with gzip.open(image_path, 'rb') as f:
                # Смещение 16 байт для заголовка IDX3
                images = np.frombuffer(f.read(), np.uint8, offset=16)
                # MNIST изображения 28x28 = 784
                images = images.reshape(-1, 784).astype(np.float32) / 255.0

            # Load labels
            with gzip.open(label_path, 'rb') as f:
                # Смещение 8 байт для заголовка IDX1
                labels = np.frombuffer(f.read(), np.uint8, offset=8)

            # Convert to one-hot if needed
            if self.one_hot:
                # Используем tf.keras.utils.to_categorical для надежности
                one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=self.class_num)
                # labels = tf.one_hot(labels, self.class_num).numpy() # Альтернатива
                return images, one_hot_labels

            return images, labels
        except Exception as e:
            print(f"Ошибка при загрузке или обработке файлов {prefix}: {e}")
            raise

# --- Словари классов (как в оригинале) ---
dict_2class = {0:'Benign',1:'Malware'}
dict_10class_benign = {0:'BitTorrent',1:'Facetime',2:'FTP',3:'Gmail',4:'MySQL',5:'Outlook',6:'Skype',7:'SMB',8:'Weibo',9:'WorldOfWarcraft'}
dict_10class_malware = {0:'Cridex',1:'Geodo',2:'Htbot',3:'Miuref',4:'Neris',5:'Nsis-ay',6:'Shifu',7:'Tinba',8:'Virut',9:'Zeus'}
dict_20class = {0:'BitTorrent', 1:'Facetime', 2:'FTP', 3:'Gmail', 4:'MySQL',
               5:'Outlook', 6:'Skype', 7:'SMB', 8:'Weibo', 9:'WorldOfWarcraft',
               10:'Cridex', 11:'Geodo', 12:'Htbot', 13:'Miuref', 14:'Neris',
               15:'Nsis-ay', 16:'Shifu', 17:'Tinba', 18:'Virut', 19:'Zeus'}

# --- Создание модели (максимально близко к оригиналу) ---
def create_original_model(class_num, include_softmax=True):
    """
    Создает модель CNN, максимально приближенную к оригинальной статье.

    Args:
        class_num (int): Количество выходных классов.
        include_softmax (bool): Включать ли Softmax в последний слой.
                                  (Установите False для экспорта в форматы,
                                   не поддерживающие Softmax).
    Returns:
        tf.keras.Model: Скомпилированная модель Keras.
    """
    model = models.Sequential(name=f"OriginalCNN_{class_num}class")
    model.add(layers.Input(shape=(784,), name='input_flat')) # Явный входной слой
    model.add(layers.Reshape((28, 28, 1), name='input_reshape'))

    # first convolutional layer (w_conv1, b_conv1, h_conv1, h_pool1)
    model.add(layers.Conv2D(32, (5, 5), padding='same', activation='relu', name='conv1'))
    # Оригинал использовал padding='SAME' и для max_pool
    model.add(layers.MaxPooling2D((2, 2), padding='same', name='pool1'))

    # second convolutional layer (w_conv2, b_conv2, h_conv2, h_pool2)
    model.add(layers.Conv2D(64, (5, 5), padding='same', activation='relu', name='conv2'))
    # Оригинал использовал padding='SAME' и для max_pool
    model.add(layers.MaxPooling2D((2, 2), padding='same', name='pool2'))

    # densely connected layer (w_fc1, b_fc1, h_pool2_flat, h_fc1)
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(1024, activation='relu', name='dense1'))

    # dropout (h_fc1_drop) - Keras автоматически управляет им
    model.add(layers.Dropout(0.5, name='dropout'))

    # readout layer (w_fc2, b_fc2, y_conv)
    if include_softmax:
        model.add(layers.Dense(class_num, activation='softmax', name='output_softmax'))
        loss_function = tf.keras.losses.CategoricalCrossentropy()
    else:
        # Вывод логитов (для экспорта в NMDL/ONNX, если Softmax не поддерживается)
        model.add(layers.Dense(class_num, name='output_logits'))
        loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Компиляция модели (оптимизатор и loss как в оригинале)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # Альтернатива: Adam часто сходится лучше

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy'])

    print("--- Модель создана ---")
    model.summary() # Выводим структуру модели
    print("----------------------")
    return model

# --- Обучение модели ---
def train_model(model, data_loader, train_steps, batch_size, model_save_path):
    """
    Обучает модель с использованием model.fit и сохраняет лучшую версию.

    Args:
        model (tf.keras.Model): Скомпилированная модель Keras.
        data_loader (MNISTDataLoader): Загрузчик данных.
        train_steps (int): Общее количество шагов обучения (как TRAIN_ROUND).
        batch_size (int): Размер батча.
        model_save_path (str): Путь для сохранения лучшей модели (.h5).

    Returns:
        tf.keras.callbacks.History: История обучения.
    """
    # Рассчитываем количество эпох
    steps_per_epoch = max(1, data_loader.num_train_examples // batch_size)
    epochs = max(1, train_steps // steps_per_epoch)
    print(f"Расчетное количество эпох: {epochs} ({train_steps} шагов / {steps_per_epoch} шагов в эпохе)")

    # Создаем директорию для сохранения, если ее нет
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Колбэк для сохранения лучшей модели по val_accuracy
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False, # Сохраняем всю модель
        mode='max',
        verbose=1
    )
    # Колбэк для ранней остановки, если улучшений нет
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10, # Количество эпох без улучшений перед остановкой (настройте)
        verbose=1,
        restore_best_weights=True # Восстановить лучшие веса в конце
    )

    print(f"Начало обучения на {epochs} эпох...")
    history = model.fit(
        data_loader.train_images,
        data_loader.train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(data_loader.test_images, data_loader.test_labels),
        callbacks=[checkpoint, early_stopping],
        verbose=1 # Показываем прогресс
    )
    print("Обучение завершено.")
    # Модель уже содержит лучшие веса благодаря restore_best_weights=True
    return history

# --- Оценка модели ---
def evaluate_model(model, data_loader, class_num, data_dir):
    """
    Оценивает модель на тестовых данных и выводит метрики.

    Args:
        model (tf.keras.Model): Обученная модель Keras.
        data_loader (MNISTDataLoader): Загрузчик данных.
        class_num (int): Количество классов.
        data_dir (str): Директория данных (для выбора словаря).
    """
    print("\nНачало оценки модели на тестовых данных...")
    y_true_one_hot = data_loader.test_labels
    y_true = np.argmax(y_true_one_hot, axis=1)

    # Получаем предсказания (логиты или вероятности, в зависимости от include_softmax)
    predictions = model.predict(data_loader.test_images)

    # Если модель выводит логиты, применяем argmax
    # Если выводит вероятности (softmax), argmax тоже сработает
    y_pred = np.argmax(predictions, axis=1)

    # Выбор словаря для имен классов
    folder = os.path.basename(data_dir)
    dict_labels = {}
    if class_num == 2: dict_labels = dict_2class
    elif class_num == 20: dict_labels = dict_20class
    elif class_num == 10:
        if folder.startswith('Benign'): dict_labels = dict_10class_benign
        elif folder.startswith('Malware'): dict_labels = dict_10class_malware
    label_names = [dict_labels.get(i, f'Class_{i}') for i in range(class_num)]

    # Расчет метрик
    accuracy = accuracy_score(y_true, y_pred)
    # average=None возвращает метрики для каждого класса
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(class_num)), zero_division=0
    )
    # Средние метрики (можно использовать 'macro' или 'weighted')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    print("--- Результаты Оценки ---")
    print(f"Общая точность (Accuracy): {accuracy:.4f}")
    print(f"Precision (Macro Avg):    {precision_macro:.4f}")
    print(f"Recall (Macro Avg):       {recall_macro:.4f}")
    print(f"F1-Score (Macro Avg):     {f1_macro:.4f}")
    print("\nМетрики по классам:")
    print(f"{'Класс':<6} {'Имя':<18} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 70)
    acc_list_str = []
    for i in range(class_num):
        name = label_names[i]
        print(f"{i:<6} {name:<18} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<8}")
        # Формируем строку для записи в файл (как в оригинале)
        acc_list_str.append([str(i), name, f"{precision[i]:.4f}", f"{recall[i]:.4f}"])

    # Запись в файл (добавляем в конец)
    try:
        with open('out_tf2.txt', 'a') as f:
            f.write("\n")
            t = time.strftime('%Y-%m-%d %X', time.localtime())
            f.write(t + "\n")
            f.write(f'DATA_DIR: {data_dir}\n')
            f.write(f'CLASS_NUM: {class_num}\n')
            f.write(f'MODEL: Original TF2 Reimplementation\n')
            f.write("Класс, Имя, Precision, Recall\n")
            for item in acc_list_str:
                f.write(', '.join(item) + "\n")
            f.write(f'Total accuracy: {accuracy:.4f}\n')
            f.write(f'Precision (Macro): {precision_macro:.4f}\n')
            f.write(f'Recall (Macro): {recall_macro:.4f}\n')
            f.write(f'F1-Score (Macro): {f1_macro:.4f}\n\n')
        print("\nРезультаты оценки записаны в out_tf2.txt")
    except Exception as e:
        print(f"\nОшибка записи результатов оценки в файл: {e}")

    print("-------------------------")

# --- Экспорт в ONNX ---
def export_to_onnx(model, onnx_save_path):
    """
    Экспортирует модель Keras в формат ONNX.

    Args:
        model (tf.keras.Model): Обученная модель Keras.
        onnx_save_path (str): Путь для сохранения ONNX модели.

    Returns:
        bool: True если экспорт успешен, иначе False.
    """
    print(f"\nЭкспорт модели в ONNX: {onnx_save_path}")
    try:
        # Определяем входную сигнатуру (плоский вектор 784)
        # Используем None для батч-размера
        input_signature = [tf.TensorSpec((None, 784), tf.float32, name='input_flat')] # Имя как у Input слоя

        # Конвертируем и сохраняем
        # opset=13 часто хорошо поддерживается
        model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
            model, input_signature, opset=13, output_path=onnx_save_path
        )
        print(f"Модель успешно экспортирована в ONNX: {onnx_save_path}")
        return True
    except Exception as e:
        print(f"Ошибка экспорта в ONNX: {e}")
        return False

# --- Проверка ONNX модели ---
def test_onnx_model(onnx_path, data_loader):
    """
    Загружает ONNX модель и выполняет инференс на нескольких примерах.

    Args:
        onnx_path (str): Путь к ONNX модели.
        data_loader (MNISTDataLoader): Загрузчик данных для тестовых примеров.
    """
    if not onnx_path or not os.path.exists(onnx_path):
        print("Пропуск проверки ONNX: файл не найден.")
        return

    print(f"\nПроверка ONNX модели: {onnx_path}")
    try:
        sess = ort.InferenceSession(onnx_path, providers=ort.get_available_providers()) # Используем доступные провайдеры
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        print(f"  ONNX Input: '{input_name}', Output: '{output_name}'")

        # Берем несколько тестовых примеров
        sample_images = data_loader.test_images[:5]
        sample_labels_true = np.argmax(data_loader.test_labels[:5], axis=1)

        # Выполняем инференс
        outputs_onnx = sess.run([output_name], {input_name: sample_images.astype(np.float32)})[0]
        predictions_onnx = np.argmax(outputs_onnx, axis=1)

        print("  Примеры предсказаний ONNX:")
        for i in range(len(sample_images)):
            print(f"    Пример {i+1}: Предсказано={predictions_onnx[i]}, Истина={sample_labels_true[i]}")

        # Сравним с Keras моделью (если она доступна глобально - не лучший стиль, но для примера)
        # predictions_keras = np.argmax(model.predict(sample_images), axis=1)
        # print(f"  Совпадают ли предсказания Keras и ONNX: {np.array_equal(predictions_keras, predictions_onnx)}")

        print("Проверка ONNX модели завершена.")

    except Exception as e:
        print(f"Ошибка проверки ONNX модели: {e}")


# --- Основной блок ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение и оценка CNN для классификации трафика (TF2 reimplementation)')
    parser.add_argument('data_dir', type=str, help='Директория с данными в формате MNIST (train/t10k .gz файлы)')
    parser.add_argument('class_num', type=int, help='Количество классов (например, 2, 10, 20)')
    parser.add_argument('train_steps', type=int, help='Общее количество шагов обучения (TRAIN_ROUND)')
    parser.add_argument('--batch_size', type=int, default=50, help='Размер батча (по умолчанию: 50)')
    parser.add_argument('--model_dir', type=str, default='models_tf2', help='Директория для сохранения моделей (по умолчанию: models_tf2)')
    parser.add_argument('--load_weights', type=str, default=None, help='Путь к файлу .h5 для загрузки весов вместо обучения')
    parser.add_argument('--skip_train', action='store_true', help='Пропустить обучение (требует --load_weights)')
    parser.add_argument('--export_onnx', action='store_true', help='Экспортировать модель в ONNX')
    parser.add_argument('--test_onnx', action='store_true', help='Проверить экспортированную ONNX модель')
    parser.add_argument('--no_softmax', action='store_true', help='Создать модель без финального Softmax слоя (для совместимости с NMDL)')

    args = parser.parse_args()

    # Проверка аргументов
    if args.skip_train and not args.load_weights:
        print("Ошибка: --skip_train требует указания --load_weights.")
        sys.exit(1)
    if args.test_onnx and not args.export_onnx:
        print("Предупреждение: --test_onnx не имеет смысла без --export_onnx. Экспорт будет выполнен.")
        args.export_onnx = True

    # Загрузка данных
    try:
        data_loader = MNISTDataLoader(args.data_dir, one_hot=True, class_num=args.class_num)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
    except Exception as e:
         print(f"Непредвиденная ошибка при загрузке данных: {e}")
         sys.exit(1)


    # Определяем пути для сохранения
    folder_name = os.path.basename(args.data_dir) or f"data_{args.class_num}class"
    model_base_name = f"model_{args.class_num}class_{folder_name}"
    if args.no_softmax:
        model_base_name += "_logits"
    model_save_dir = os.path.join(args.model_dir, model_base_name)
    model_save_path = os.path.join(model_save_dir, model_base_name + ".h5")
    onnx_save_path = os.path.join(model_save_dir, model_base_name + ".onnx")

    # Создание модели
    model = create_original_model(args.class_num, include_softmax=(not args.no_softmax))

    # Обучение или загрузка весов
    if not args.skip_train:
        if args.load_weights:
            if os.path.exists(args.load_weights):
                print(f"Загрузка весов из: {args.load_weights}")
                model.load_weights(args.load_weights)
            else:
                print(f"Предупреждение: Файл весов {args.load_weights} не найден. Начинаем обучение с нуля.")
                train_model(model, data_loader, args.train_steps, args.batch_size, model_save_path)
        else:
            train_model(model, data_loader, args.train_steps, args.batch_size, model_save_path)
            print(f"Лучшая модель сохранена в: {model_save_path}")
            # Загружаем лучшую модель после обучения для последующих шагов
            print("Загрузка лучшей сохраненной модели для оценки/экспорта...")
            model = tf.keras.models.load_model(model_save_path) # Загружаем всю модель

    elif args.load_weights: # Если пропустили трейн, но указали веса
         if os.path.exists(args.load_weights):
             print(f"Загрузка весов из: {args.load_weights}")
             # Загружаем только веса в созданную архитектуру
             model.load_weights(args.load_weights)
             print("Веса загружены.")
         else:
             print(f"Ошибка: Файл весов {args.load_weights} не найден, обучение пропущено.")
             sys.exit(1)


    # Оценка модели
    evaluate_model(model, data_loader, args.class_num, args.data_dir)

    # Экспорт в ONNX
    onnx_exported = False
    if args.export_onnx:
       os.makedirs(model_save_dir, exist_ok=True) # Убедимся что директория есть
       onnx_exported = export_to_onnx(model, onnx_save_path)

    # Проверка ONNX
    if args.test_onnx and onnx_exported:
        test_onnx_model(onnx_save_path, data_loader)
    elif args.test_onnx and not onnx_exported:
        print("Пропуск проверки ONNX: экспорт не удался или был пропущен.")

    print("\nСкрипт завершен.")