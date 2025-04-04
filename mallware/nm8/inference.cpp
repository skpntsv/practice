#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>   // Для std::accumulate, std::exp
#include <algorithm> // Для std::max_element, std::distance
#include <map>       // Для удобного хранения имен классов (хотя вектор тоже подойдет)
#include <iomanip>   // Для форматирования вывода (std::fixed, std::setprecision)
#include <cmath>     // Для std::exp
#include <thread>

#include "nmdl.h"
// Заголовочные файлы компилятора и конвертера изображений не нужны для инференса
// #include "nmdl_compiler.h"
// #include "nmdl_image_converter.h"

// --- Ваши функции Call и ReadFile остаются без изменений ---

// Функция для обработки возвращаемых значений NMDL (оставляем как есть)
NMDL_RESULT Call(NMDL_RESULT result, const std::string &function_name) {
    switch(result) {
        case NMDL_RESULT_OK:
            return NMDL_RESULT_OK;
        case NMDL_RESULT_INVALID_FUNC_PARAMETER:
            throw std::runtime_error(function_name + ": INVALID_FUNC_PARAMETER");
        case NMDL_RESULT_NO_LOAD_LIBRARY:
            throw std::runtime_error(function_name + ": NO_LOAD_LIBRARY");
        case NMDL_RESULT_NO_BOARD:
            throw std::runtime_error(function_name + ": NO_BOARD");
        case NMDL_RESULT_BOARD_RESET_ERROR:
            throw std::runtime_error(function_name + ": BOARD_RESET_ERROR");
        case NMDL_RESULT_INIT_CODE_LOADING_ERROR:
            throw std::runtime_error(function_name + ": INIT_CODE_LOADING_ERROR");
        case NMDL_RESULT_CORE_HANDLE_RETRIEVAL_ERROR:
            throw std::runtime_error(function_name + ": CORE_HANDLE_RETRIEVAL_ERROR");
        case NMDL_RESULT_FILE_LOADING_ERROR:
            throw std::runtime_error(function_name + ": FILE_LOADING_ERROR");
        case NMDL_RESULT_MEMORY_WRITE_ERROR:
            throw std::runtime_error(function_name + ": MEMORY_WRITE_ERROR");
        case NMDL_RESULT_MEMORY_READ_ERROR:
            throw std::runtime_error(function_name + ": MEMORY_READ_ERROR");
        case NMDL_RESULT_MEMORY_ALLOCATION_ERROR:
            throw std::runtime_error(function_name + ": MEMORY_ALLOCATION_ERROR");
        case NMDL_RESULT_MODEL_LOADING_ERROR:
            throw std::runtime_error(function_name + ": MODEL_LOADING_ERROR");
        case NMDL_RESULT_INVALID_MODEL:
            throw std::runtime_error(function_name + ": INVALID_MODEL");
        case NMDL_RESULT_BOARD_SYNC_ERROR:
            throw std::runtime_error(function_name + ": BOARD_SYNC_ERROR");
        case NMDL_RESULT_BOARD_MEMORY_ALLOCATION_ERROR:
            throw std::runtime_error(function_name + ": BOARD_MEMORY_ALLOCATION_ERROR");
        case NMDL_RESULT_NN_CREATION_ERROR:
            throw std::runtime_error(function_name + ": NN_CREATION_ERROR");
        case NMDL_RESULT_NN_LOADING_ERROR:
            throw std::runtime_error(function_name + ": NN_LOADING_ERROR");
        case NMDL_RESULT_NN_INFO_RETRIEVAL_ERROR:
            throw std::runtime_error(function_name + ": NN_INFO_RETRIEVAL_ERROR");
        case NMDL_RESULT_MODEL_IS_TOO_BIG:
            throw std::runtime_error(function_name + ": MODEL_IS_TOO_BIG");
        case NMDL_RESULT_NOT_INITIALIZED:
            throw std::runtime_error(function_name + ": NOT_INITIALIZED");
        case NMDL_RESULT_INCOMPLETE:
            throw std::runtime_error(function_name + ": INCOMPLETE");
        case NMDL_RESULT_UNKNOWN_ERROR:
            throw std::runtime_error(function_name + ": UNKNOWN_ERROR");
        default:
            throw std::runtime_error(function_name + ": UNKNOWN ERROR");
    }
}

// Универсальная функция для чтения файла в вектор (оставляем как есть)
template <typename T>
std::vector<T> ReadFile(const std::string &filename) {
    // ... (ваш код без изменений) ...
     std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if(!ifs.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    auto fsize = static_cast<std::size_t>(ifs.tellg());
    if (fsize == 0) {
         throw std::runtime_error("Input file is empty: " + filename);
    }
    // Проверка, делится ли размер файла на размер элемента
    if (fsize % sizeof(T) != 0) {
         throw std::runtime_error("File size " + std::to_string(fsize) + " is not a multiple of element size " + std::to_string(sizeof(T)) + " in file: " + filename);
    }
    ifs.seekg(0);
    std::vector<T> data(fsize / sizeof(T));
    ifs.read(reinterpret_cast<char*>(data.data()), fsize); // Читаем fsize байт
     if (!ifs) {
        throw std::runtime_error("Error reading file: " + filename);
    }
    return data;
}


// Вывод версии библиотеки
void ShowNMDLVersion() {
    std::uint32_t major = 0, minor = 0, patch = 0;
    Call(NMDL_GetLibVersion(&major, &minor, &patch), "GetLibVersion");
    std::cout << "Lib version: " << major << "." << minor << "." << patch << std::endl;
}

// Проверка наличия плат
void CheckBoard(std::uint32_t board_type) {
    std::uint32_t boards = 0;
    Call(NMDL_GetBoardCount(board_type, &boards), "GetBoardCount");
    std::cout << "Detected boards: " << boards << std::endl;
    if(boards == 0) {
        throw std::runtime_error("Board not found");
    }
}

// Получение информации о модели
NMDL_ModelInfo GetModelInformation(NMDL_HANDLE nmdl, std::uint32_t unit_num) {
    NMDL_ModelInfo model_info;
    Call(NMDL_GetModelInfo(nmdl, unit_num, &model_info), "GetModelInfo");
    std::cout << "Input tensor number: " << model_info.input_tensor_num << std::endl;
    for(std::size_t i = 0; i < model_info.input_tensor_num; ++i) {
        std::cout << "Input tensor " << i << ": " 
                  << model_info.input_tensors[i].width << " x " 
                  << model_info.input_tensors[i].height << " x " 
                  << model_info.input_tensors[i].depth << std::endl;
    }
    std::cout << "Output tensor number: " << model_info.output_tensor_num << std::endl;
    for(std::size_t i = 0; i < model_info.output_tensor_num; ++i) {
        std::cout << "Output tensor " << i << ": " 
                  << model_info.output_tensors[i].width << " x " 
                  << model_info.output_tensors[i].height << " x " 
                  << model_info.output_tensors[i].depth << std::endl;
    }
    return model_info;
}


// --- НОВЫЕ ФУНКЦИИ ---

// Имена классов (ВАЖНО: порядок должен точно соответствовать вашему обучению!)
const std::vector<std::string> CLASS_NAMES = {
    "BitTorrent", "Facetime", "FTP", "Gmail", "MySQL",
    "Outlook", "Skype", "SMB", "Weibo", "WorldOfWarcraft",
    "Cridex", "Geodo", "Htbot", "Miuref", "Neris",
    "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus"
};

// Применение Softmax к вектору логитов для получения вероятностей
std::vector<float> apply_softmax(const std::vector<float>& logits) {
    if (logits.empty()) {
        return {};
    }
    std::vector<float> probabilities;
    probabilities.reserve(logits.size());

    // Находим максимальный логит для численной стабильности
    float max_logit = *std::max_element(logits.begin(), logits.end());

    double sum_exp = 0.0;
    for (float logit : logits) {
        // Вычитаем максимум перед экспонентой
        double exp_val = std::exp(static_cast<double>(logit) - max_logit);
        probabilities.push_back(static_cast<float>(exp_val));
        sum_exp += exp_val;
    }

    // Нормализуем на сумму экспонент
    if (sum_exp > 1e-9) { // Избегаем деления на ноль
        for (float& prob : probabilities) {
            prob /= static_cast<float>(sum_exp);
        }
    } else {
        // Если сумма очень мала, можем вернуть равномерное распределение или оставить как есть (нули)
         std::fill(probabilities.begin(), probabilities.end(), 1.0f / probabilities.size());
    }

    return probabilities;
}

void PrintRawLogits(const std::vector<float>& output_logits, size_t frame_index) {
    std::cout << "Frame " << frame_index << " Raw Logits (" << output_logits.size() << " values):" << std::endl;
    std::cout << std::fixed << std::setprecision(6); // Устанавливаем точность вывода
    for (size_t i = 0; i < output_logits.size(); ++i) {
        std::cout << "  [" << std::setw(2) << i << "] "; // Выводим индекс
        if (i < CLASS_NAMES.size()) {
             std::cout << std::left << std::setw(18) << CLASS_NAMES[i] << ": "; // Выводим имя класса, если есть
        } else {
             std::cout << std::left << std::setw(18) << "Unknown" << ": ";
        }
         std::cout << std::right << std::setw(12) << output_logits[i] << std::endl; // Выводим значение логита
    }
    std::cout << std::endl; // Пустая строка для разделения
}

// Обработка выходного тензора и вывод результата
void ProcessOutput(const std::vector<float>& output_logits, size_t frame_index) {
    PrintRawLogits(output_logits, frame_index);

    // if (output_logits.size() != CLASS_NAMES.size()) {
    //     std::cerr << "Warning: Output tensor size (" << output_logits.size()
    //               << ") does not match number of class names (" << CLASS_NAMES.size() << ")" << std::endl;
    //     // Выводим сырые логиты, если размеры не совпадают
    //     std::cout << "Frame " << frame_index << " Raw Logits (first 10): ";
    //      for(size_t i = 0; i < std::min((size_t)10, output_logits.size()); ++i) {
    //          std::cout << output_logits[i] << " ";
    //      }
    //      std::cout << "..." << std::endl;
    //     return;
    // }

    // // 1. Применяем Softmax
    // std::vector<float> probabilities = apply_softmax(output_logits);

    // // 2. Находим индекс максимальной вероятности
    // auto max_it = std::max_element(probabilities.begin(), probabilities.end());
    // int predicted_index = std::distance(probabilities.begin(), max_it);
    // float confidence = *max_it;

    // // 3. Получаем имя класса
    // std::string predicted_class_name = (predicted_index >= 0 && predicted_index < CLASS_NAMES.size())
    //                                    ? CLASS_NAMES[predicted_index]
    //                                    : "Unknown Index";

    // // 4. Выводим результат
    // std::cout << "Frame " << frame_index << " Prediction: Class '" << predicted_class_name
    //           << "' (Index " << predicted_index << ") with confidence "
    //           << std::fixed << std::setprecision(4) << confidence << std::endl;

    // --- Дополнительно: Вывести топ-N предсказаний ---
    // const int top_n = 3;
    // std::vector<std::pair<float, int>> sorted_probs;
    // for(int i=0; i < probabilities.size(); ++i) {
    //     sorted_probs.push_back({probabilities[i], i});
    // }
    // std::sort(sorted_probs.rbegin(), sorted_probs.rend()); // Сортируем по убыванию вероятности
    // std::cout << "  Top " << top_n << " predictions:" << std::endl;
    // for(int i=0; i < std::min(top_n, (int)sorted_probs.size()); ++i) {
    //     int idx = sorted_probs[i].second;
    //     std::string name = (idx >= 0 && idx < CLASS_NAMES.size()) ? CLASS_NAMES[idx] : "Unknown";
    //     std::cout << "    - " << name << " (" << idx << "): "
    //               << std::fixed << std::setprecision(4) << sorted_probs[i].first << std::endl;
    // }
}


// --- Функция ожидания WaitForOutput ---
// Модифицируем, чтобы она не выводила первые 4 значения,
// а просто возвращала FPS после получения результата.
double WaitForOutput(NMDL_HANDLE nmdl, std::uint32_t unit_num, float *outputs[], unsigned int timeout_ms = 10000) { // Увеличен таймаут
    std::uint32_t status = NMDL_PROCESS_FRAME_STATUS_INCOMPLETE;
    auto start_wait = std::chrono::steady_clock::now();
    bool status_checked = false; // Флаг для однократного вывода статуса ожидания

    while (status == NMDL_PROCESS_FRAME_STATUS_INCOMPLETE) {
        // Периодически проверяем статус
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Небольшая пауза

        NMDL_RESULT status_res = NMDL_GetStatus(nmdl, unit_num, &status);
        if (status_res != NMDL_RESULT_OK && status_res != NMDL_RESULT_INCOMPLETE) {
             // Если GetStatus вернул ошибку (кроме INCOMPLETE), пробрасываем ее
            Call(status_res, "GetStatus");
        }
        status_checked = true;

        // Проверка таймаута
        auto now = std::chrono::steady_clock::now();
        unsigned int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_wait).count();
        if (elapsed > timeout_ms) {
            throw std::runtime_error("Timeout (" + std::to_string(timeout_ms) + "ms) while waiting for output on unit " + std::to_string(unit_num));
        }
    }

     if (!status_checked) { // Если цикл ни разу не выполнился (статус сразу OK)
        // Все равно вызовем GetStatus один раз для консистентности, хотя статус уже известен
        Call(NMDL_GetStatus(nmdl, unit_num, &status), "GetStatus Check After Loop");
     }


    // Статус != NMDL_PROCESS_FRAME_STATUS_INCOMPLETE, можно получать результат
    double fps = 0.0;
    Call(NMDL_GetOutput(nmdl, unit_num, outputs, &fps), "GetOutput");
    // Убираем вывод первых 4 значений отсюда
    // std::cout << "FPS reported by NMDL_GetOutput: " << fps << std::endl; // Можно оставить для отладки
    return fps;
}


// --- Модифицированная функция main ---
int main(int argc, char *argv[]) {
    // --- Параметры ---
    // Используем NMDL_BOARD_TYPE_NMCARD для NM Card Mini (соответствует чипу K1879BM8Я)
    // Смотрите руководство NMDL для точного соответствия вашей карты
    const std::uint32_t BOARD_TYPE = NMDL_BOARD_TYPE_NMCARD;

    // Имена файлов можно передавать через аргументы командной строки для гибкости
    std::string model_filename = "model_20class_SessionAllLayers_logits.nm8"; // Ваша скомпилированная модель БЕЗ Softmax
    std::string input_filename = "prepared_input.bin";      // Имя файла с подготовленным входным тензором
    std::size_t frames_to_process = 1;                      // Количество раз для обработки входного файла

    if (argc > 1) model_filename = argv[1];
    if (argc > 2) input_filename = argv[2];
    if (argc > 3) {
        try {
            frames_to_process = std::stoul(argv[3]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid frame count argument: " << argv[3] << ". Using default: " << frames_to_process << std::endl;
        }
    }

    std::cout << "Using Model: " << model_filename << std::endl;
    std::cout << "Using Input: " << input_filename << std::endl;
    std::cout << "Processing Count: " << frames_to_process << std::endl;
    // --- Конец Параметров ---


    NMDL_HANDLE nmdl = 0; // Инициализируем нулем

    try {
        std::cout << "Query library version..." << std::endl;
        ShowNMDLVersion();

        std::cout << "Board detection (Type: " << BOARD_TYPE << ")..." << std::endl;
        CheckBoard(BOARD_TYPE); // Проверяем наличие нужного типа платы

        std::cout << "NMDL initialization..." << std::endl;
        Call(NMDL_Create(&nmdl), "Create");

        // Загрузка модели
        std::cout << "Loading model from file: " << model_filename << std::endl;
        // Модель .nm8 - это бинарные данные, но NMDL_Initialize ожидает float*
        // Читаем как char, затем reinterpret_cast, или убедимся, что ReadFile<float> работает корректно
        // Предполагаем, что модель .nm8 содержит данные float или может быть прочитана как float
         std::vector<float> model_data;
         try {
            model_data = ReadFile<float>(model_filename);
         } catch (const std::runtime_error& e) {
            std::cerr << "Error loading model file: " << e.what() << std::endl;
            throw; // Пробрасываем исключение дальше
         }

        std::cout << "Model size: " << model_data.size() << " float elements (" << model_data.size() * sizeof(float) << " bytes)" << std::endl;
        // NMDL_Initialize ожидает массив указателей и массив размеров
        std::array<const float*, NMDL_MAX_UNITS> models_ptr = { model_data.data() }; // Указатель на данные модели
        std::array<std::uint32_t, NMDL_MAX_UNITS> model_sizes_float = { static_cast<std::uint32_t>(model_data.size()) }; // Размер в float'ах

        // Инициализация NMDL (используем unit 0, режим single unit)
        std::cout << "Initializing NMDL device with the model..." << std::endl;
        // Последний аргумент (use_batch_mode) не показан в вашем примере Initialize,
        // возможно, он в старой версии API или не требуется для NMCARD. Используем API из примера.
        Call(NMDL_Initialize(nmdl, BOARD_TYPE, 0, 0, models_ptr.data(), model_sizes_float.data()), "Initialize");
        std::cout << "NMDL Initialized." << std::endl;

        // Получаем информацию о модели (размеры входа/выхода)
        std::cout << "Getting model information..." << std::endl;
        NMDL_ModelInfo model_info = GetModelInformation(nmdl, 0); // Получаем инфо для unit 0

        // Проверяем, что у модели один вход и один выход (как ожидается)
         if (model_info.input_tensor_num != 1 || model_info.output_tensor_num != 1) {
             throw std::runtime_error("Model has unexpected number of inputs/outputs: inputs="
                                      + std::to_string(model_info.input_tensor_num) + ", outputs="
                                      + std::to_string(model_info.output_tensor_num));
         }
         // Проверяем размерность входного тензора (должен быть 784)
         size_t expected_input_size = model_info.input_tensors[0].width *
                                    //  model_info.input_tensors[0].height *
                                     model_info.input_tensors[0].depth;
         if (expected_input_size != 784) {
              throw std::runtime_error("Model expects input size " + std::to_string(expected_input_size)
                                       + ", but preprocessing assumes 784.");
         }
         // Проверяем размерность выходного тензора (должен быть = class_num)
         size_t output_size = model_info.output_tensors[0].width *
                             model_info.output_tensors[0].height *
                             model_info.output_tensors[0].depth;
         if (output_size != CLASS_NAMES.size()) {
              throw std::runtime_error("Model output size " + std::to_string(output_size)
                                       + " does not match CLASS_NAMES size " + std::to_string(CLASS_NAMES.size()));
         }


        // Загрузка подготовленного входного тензора
        std::cout << "Loading input tensor from file: " << input_filename << std::endl;
        std::vector<float> input_tensor_data;
        try {
            input_tensor_data = ReadFile<float>(input_filename);
        } catch (const std::runtime_error& e) {
             std::cerr << "Error loading input file: " << e.what() << std::endl;
             throw;
        }


        // Проверка размера входного тензора
        if (input_tensor_data.size() != expected_input_size) {
            throw std::runtime_error("Input file '" + input_filename + "' size (" + std::to_string(input_tensor_data.size())
                                     + ") does not match model expected input size (" + std::to_string(expected_input_size) + ")");
        }
        // NMDL_Process ожидает массив указателей на входы
        std::array<const float*, 1> inputs_ptr = { input_tensor_data.data() };


        // Выделение памяти для выходных данных (только один выходной тензор)
        std::cout << "Reserving output buffers..." << std::endl;
        std::vector<float> output_tensor_data(output_size);
        // NMDL_GetOutput ожидает массив указателей на выходы
        std::array<float*, 1> outputs_ptr = { output_tensor_data.data() };
        std::cout << "Allocated " << output_size << " float elements for output tensor 0" << std::endl;


        // Запуск инференса
        std::cout << "Processing " << frames_to_process << " times with input '" << input_filename << "'..." << std::endl;
        double total_fps = 0.0;
        auto total_start_time = std::chrono::high_resolution_clock::now();

        for(std::size_t frame = 0; frame < frames_to_process; ++frame) {
            // std::cout << "Processing frame " << frame + 1 << "/" << frames_to_process << "..." << std::endl;
            auto frame_start_time = std::chrono::high_resolution_clock::now();

            // Запускаем обработку на unit 0
            Call(NMDL_Process(nmdl, 0, inputs_ptr.data()), "Process frame " + std::to_string(frame));

            // Ждем результат и получаем FPS, сообщенный NMDL
            double current_fps = WaitForOutput(nmdl, 0, outputs_ptr.data());
            total_fps += current_fps; // NMDL сам считает FPS, но можно и вручную

            // Обрабатываем и выводим результат
            ProcessOutput(output_tensor_data, frame + 1); // Передаем вектор с результатами

            auto frame_end_time = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end_time - frame_start_time);
            // std::cout << "  Frame processing time: " << frame_duration.count() / 1000.0 << " ms" << std::endl;
             // Добавим небольшую паузу, если обработка идет слишком быстро,
             // чтобы избежать перегрузки или для демонстрации
             // std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

        std::cout << "\nProcessed " << frames_to_process << " frames." << std::endl;
        if (frames_to_process > 0) {
            std::cout << "Average FPS reported by NMDL: " << (total_fps / frames_to_process) << std::endl;
            std::cout << "Total processing time: " << total_duration.count() / 1000.0 << " s" << std::endl;
             if (total_duration.count() > 0) {
                 std::cout << "Calculated Average FPS: " << (frames_to_process * 1000.0 / total_duration.count()) << std::endl;
             }
        }

    } // Конец try
    catch (const std::exception &e) {
        std::cerr << "\n--- ERROR --- \n" << e.what() << "\n-------------" << std::endl;
        // Освобождение ресурсов в случае ошибки
        if(nmdl) {
            std::cout << "Releasing NMDL resources due to error..." << std::endl;
            NMDL_Release(nmdl); // Игнорируем ошибки здесь
            NMDL_Destroy(nmdl); // Игнорируем ошибки здесь
        }
        return 1; // Возвращаем код ошибки
    }

    // Нормальное завершение: освобождение ресурсов
    if (nmdl != 0) {
        std::cout << "\nReleasing NMDL resources..." << std::endl;
        // Можно обернуть в try-catch на случай редких ошибок при освобождении,
        // но прямой вызов здесь стандартен.
        NMDL_Release(nmdl); // Вызываем напрямую
        NMDL_Destroy(nmdl); // Вызываем напрямую
        std::cout << "Resources released." << std::endl;
    }
    std::cout << "Exiting successfully." << std::endl;
    return 0; // Успешное завершение
}