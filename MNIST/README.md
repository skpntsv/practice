# NeuroMatrix MNIST Demo

Этот проект демонстрирует полный конвейер переноса простой сверточной сети для распознавания MNIST на платформу NeuroMatrix® (симулятор MC127.05). Мы реализовали:

1. **Обучение** и **экспорт** модели в ONNX-формате (Jupyter Notebook + PyTorch).  
2. **Локальную проверку** ONNX-модели на том же тестовом образце.  
3. **Конвертацию** ONNX→NM8 (NMDL Compiler) и **инференс** в C++ на симуляторе NeuroMatrix.  
4. **Сравнение** выходов ONNX Runtime и NeuroMatrix-симулятора.

В будущем этот пример можно взять за основу для переноса более сложных сетей.

---

## Основные нюансы

- **Форма входа**  
  Наша модель обучена на MNIST (1×28×28), ожидает тензор(shape=`[B,1,28,28]`) с пикселями в диапазоне **[0,1]**.

- **Нормализация**  
  - В Python: `arr = np.array(img, dtype=float32) / 255.0`.  
  - В NMDL Image Converter: устанавливаем  
    ```cpp
    COLOR_FMT = NMDL_IMAGE_CONVERTER_COLOR_FORMAT_INTENSITY;
    rgb_divider = {255.0f, 0, 0}; // только первый компонент важен
    rgb_adder   = {0.0f,   0, 0};
    ```
    чтобы `dst = src_gray / 255.0`.

- **Палитра BMP vs RAW**  
  - Мы сохраняем `sample.bmp` как 8-битный grayscale BMP (режим `'L'`), без палитры.  
  - Альтернативно можно подавать «сырые» float32 данные из `.raw` (без BMP-конвертера).

- **Compile ONNX → NM8**  
  - Используем `NMDL_COMPILER_CompileONNX(is_multi_unit=false, board_type=MC12705, …)`.  
  - Передаем `NMDL_MAX_UNITS`-длины массивы (остальные элементы = nullptr/0).

- **Инициализация**  

```cpp
  NMDL_Initialize(
    nmdl_handle,
    NMDL_BOARD_TYPE_SIMULATOR,
    /*board_number=*/0,
    /*proc_number=*/0,
    model_ptrs,    // float const*[NMDL_MAX_UNITS]
    model_sizes    // uint32_t[NMDL_MAX_UNITS]
  );
````

* **Batch vs Single-unit**
  Для демонстрации используем `is_multi_unit=false` и `FRAMES=1`. Batch-режим в примере не задействован.

---

## Установка и запуск

### 1. Клонирование и зависимости

```bash
git clone <repo_url> project-root
cd project-root
```

Установите зависимости для Python:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


### 2. Обучение и экспорт ONNX

1. Откройте `notebook/train_mnist.ipynb` в VS Code или Jupyter.
2. Запустите ячейки:

   1. Импорт и загрузка MNIST
   2. Определение CNN
   3. Тренировка (5 эпох)
   4. Проверка точности
   5. Экспорт `model/model.onnx`

### 3. Генерация тестового образца

```bash
cd data
python ./make_sample.py
# → создаст sample.bmp (28×28, grayscale)
```

### 4. Локальная проверка ONNX

1. Откройте `notebook/train_mnist.ipynb` в VS Code или Jupyter.
2. Запустите ячейку **Локальная проверка ONNX**

Вы должны увидеть 10-мерный вектор логитов, максимум на позиции метки (0…9).

### 5. Сборка и запуск C++ инференса

```bash
cd cpp
chmod +x build.sh
./build.sh
# бинарник скопируется в cpp/bin/test (или ваш BIN_DIR)
cd bin
./test
```

Вы получите похожие логиты в режиме симулятора

# Установка NMDL SDK на Ubuntu Linux

## 1. Скачивание SDK

1. Зарегистрируйтесь или войдите на сайт **NeuroMatrix** (обычно [https://module.ru](https://www.module.ru/products/4-programmno-apparatnye-kompleksy-pak/neuromatrix-deep-learning) или портал вашей организации) и скачайте **NMDL SDK** для Linux.

---

## 2. Установка

```bash
sudo -i dpkg NMDL-6.0.0-Linux-amd64.deb
```

# Проверьте содержимое

```bash
ls /opt/nmdl
# ├── bin/       — утилиты (nmdl_compiler_console, nmdl_image_converter и т.д.)
# ├── include/   — заголовочные файлы
# └── lib/       — статические и динамические библиотеки
```

---

## 3. Настройка переменных окружения

Добавьте в конец `~/.bashrc` (или `~/.profile`):

```bash
# NeuroMatrix NMDL SDK
export NMDL_ROOT=/opt/nmdl
export PATH=$NMDL_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$NMDL_ROOT/lib:$LD_LIBRARY_PATH
```

Примените изменения:

```bash
source ~/.bashrc
```

