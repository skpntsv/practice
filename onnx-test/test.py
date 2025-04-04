import onnxruntime as ort
import numpy as np
import time
import traceback

# --- Параметры ---
# Укажите путь к вашему ONNX файлу
# onnx_model_path = "./trained_models_original_like/model_20class_SessionAllLayers_logits_seq_origlike_nodrop/model_20class_SessionAllLayers_logits_seq_origlike_nodrop.onnx"
onnx_model_path = "./model.onnx" # Пример для Functional модели

# Подготовленные входные данные из Шага 2
# input_data = ... (должен быть NumPy массив правильной формы и типа)
# Используем вариант с чтением из .bin для примера
input_bin_file = "prepared_input.bin"

CLASS_NAMES = {0:'BitTorrent', 1:'Facetime', 2:'FTP', 3:'Gmail', 4:'MySQL',
               5:'Outlook', 6:'Skype', 7:'SMB', 8:'Weibo', 9:'WorldOfWarcraft',
               10:'Cridex', 11:'Geodo', 12:'Htbot', 13:'Miuref', 14:'Neris',
               15:'Nsis-ay', 16:'Shifu', 17:'Tinba', 18:'Virut', 19:'Zeus'}


try:
    flat_data = np.fromfile(input_bin_file, dtype=np.float32)
    if flat_data.shape[0] == 784:
        input_data = flat_data.reshape(1, 784).astype(np.float32) # <--- ВАЖНО: Форма (1, 784)
        print(f"Данные из {input_bin_file} загружены. Форма: {input_data.shape}")
    else:
        raise ValueError(f"Ожидалось 784 float, найдено {flat_data.shape[0]}")
except Exception as e:
    print(f"Ошибка подготовки данных: {e}")
    exit()
# ----------------

# --- Запуск ONNX Runtime ---
try:
    print(f"Загрузка ONNX модели: {onnx_model_path}")
    # Создаем сессию InferenceSession
    # providers - список провайдеров выполнения (CPU, CUDA, TensorRT и т.д.)
    # ONNX Runtime автоматически выберет лучший доступный
    available_providers = ort.get_available_providers()
    print(f"Доступные провайдеры: {available_providers}")
    # Приоритет: CUDA (если есть), потом CPU
    preferred_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    valid_providers = [p for p in preferred_providers if p in available_providers]
    if not valid_providers:
        valid_providers = ['CPUExecutionProvider'] # Гарантируем хотя бы CPU
    print(f"Используемые провайдеры: {valid_providers}")

    sess = ort.InferenceSession(onnx_model_path, providers=valid_providers)

    # Получаем имена входа и выхода
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"Имя входа ONNX: '{input_name}'")
    print(f"Имя выхода ONNX: '{output_name}'")
    print(f"Ожидаемая форма входа (от ONNX): {sess.get_inputs()[0].shape}") # Полезно для проверки

    # Выполняем инференс
    # Вход должен быть словарем {имя_входа: numpy_массив}
    print(f"Выполнение инференса...")
    start_time = time.time()
    outputs = sess.run([output_name], {input_name: input_data})
    end_time = time.time()
    print(f"Инференс завершен за {(end_time - start_time)*1000:.2f} мс")

    # Результат - это список выходных тензоров (в нашем случае один)
    output_tensor = outputs[0] # Получаем первый (и единственный) выходной тензор
    print(f"Форма выходного тензора: {output_tensor.shape}") # Должна быть (batch_size, class_num), например (1, 20)

    # Обрабатываем результат (например, находим предсказанный класс)
    # Если модель выводит логиты (как мы и делали):
    predicted_class_index = np.argmax(output_tensor, axis=1)
    # Применяем Softmax для получения вероятностей (если нужно)
    # probabilities = tf.nn.softmax(output_tensor).numpy() # Нужен TensorFlow
    # Или напишем свою softmax функцию
    def softmax_np(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    probabilities = softmax_np(output_tensor)
    predicted_confidence = np.max(probabilities, axis=1)

    # Выводим результат для каждого примера в батче
    for i in range(output_tensor.shape[0]): # Итерируем по батчу
        class_idx = predicted_class_index[i]
        confidence = predicted_confidence[i]
        # Получаем имя класса из вашего словаря или списка
        class_name = CLASS_NAMES[class_idx] if 0 <= class_idx < len(CLASS_NAMES) else "Неизвестный индекс"

        print(f"\n--- Результат для примера {i+1} ---")
        print(f"Предсказанный класс: {class_name} (Индекс: {class_idx})")
        print(f"Уверенность (Confidence): {confidence:.4f}")
        print(f"Сырые логиты: {output_tensor[i]}...")
        print(f"Вероятности: {probabilities[i]}...")

except Exception as e:
    print(f"Ошибка при работе с ONNX моделью: {e}")
    print(traceback.format_exc())