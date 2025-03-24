#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "nmdl.h"
#include "nmdl_compiler.h"
#include "nmdl_image_converter.h"

// Функция для обработки возвращаемых значений компилятора
NMDL_RESULT Call(NMDL_COMPILER_RESULT result, const std::string &function_name) {
    switch(result) {
    case NMDL_COMPILER_RESULT_OK:
        return NMDL_RESULT_OK;
    case NMDL_COMPILER_RESULT_MEMORY_ALLOCATION_ERROR:
        throw std::runtime_error(function_name + ": MEMORY_ALLOCATION_ERROR");
    case NMDL_COMPILER_RESULT_MODEL_LOADING_ERROR:
        throw std::runtime_error(function_name + ": MODEL_LOADING_ERROR");
    case NMDL_COMPILER_RESULT_INVALID_PARAMETER:
        throw std::runtime_error(function_name + ": INVALID_PARAMETER");
    case NMDL_COMPILER_RESULT_INVALID_MODEL:
        throw std::runtime_error(function_name + ": INVALID_MODEL");
    case NMDL_COMPILER_RESULT_UNSUPPORTED_OPERATION:
        throw std::runtime_error(function_name + ": UNSUPPORTED_OPERATION");
    default:
        throw std::runtime_error(function_name + ": UNKNOWN ERROR");
    }
}

// Функция для обработки возвращаемых значений NMDL
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

// Универсальная функция для чтения файла в вектор
template <typename T>
std::vector<T> ReadFile(const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if(!ifs.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    auto fsize = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0);
    std::vector<T> data(fsize / sizeof(T));
    ifs.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(T));
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

// Функция ожидания завершения обработки с таймаутом (например, 5 секунд)
double WaitForOutput(NMDL_HANDLE nmdl, std::uint32_t unit_num, float *outputs[], unsigned int timeout_ms = 5000) {
    std::uint32_t status = NMDL_PROCESS_FRAME_STATUS_INCOMPLETE;
    auto start = std::chrono::steady_clock::now();
    while(status == NMDL_PROCESS_FRAME_STATUS_INCOMPLETE) {
        Call(NMDL_GetStatus(nmdl, unit_num, &status), "GetStatus");
        auto now = std::chrono::steady_clock::now();
        unsigned int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        // if (elapsed > timeout_ms) {
        //     throw std::runtime_error("Timeout while waiting for output on unit " + std::to_string(unit_num));
        // }
    }
    double fps = 0.0;
    Call(NMDL_GetOutput(nmdl, unit_num, outputs, &fps), "GetOutput");
    std::cout << "First four result values:" << std::endl;
    for(std::size_t i = 0; i < 4; ++i) {
        std::cout << outputs[0][i] << std::endl;
    }
    std::cout << "FPS: " << fps << std::endl;
    return fps;
}

int main() {
    // Параметры: используем NM Card Mini (для которого BOARD_TYPE = NMDL_BOARD_TYPE_NMCARD)
    const std::uint32_t BOARD_TYPE = NMDL_BOARD_TYPE_NMCARD;
    const std::string MODEL_FILENAME = "model.nm8";  // скомпилированная модель вашей нейросети
    const std::string FRAME_FILENAME = "prepared_input";  // подготовленный входной тензор
    const std::size_t FRAMES = 10;  // количество обрабатываемых кадров

    NMDL_HANDLE nmdl = 0;

    try {
        std::cout << "Query library version..." << std::endl;
        ShowNMDLVersion();

        std::cout << "Board detection..." << std::endl;
        CheckBoard(BOARD_TYPE);

        std::cout << "NMDL initialization..." << std::endl;
        Call(NMDL_Create(&nmdl), "Create");

        // Загрузка модели
        std::cout << "Loading model from file: " << MODEL_FILENAME << std::endl;
        std::vector<float> model = ReadFile<float>(MODEL_FILENAME);
        std::array<const float*, NMDL_MAX_UNITS> models = { model.data() };
        std::array<std::uint32_t, NMDL_MAX_UNITS> model_floats = { static_cast<std::uint32_t>(model.size()) };

        // Инициализация NMDL (режим single unit)
        Call(NMDL_Initialize(nmdl, BOARD_TYPE, 0, 0, models.data(), model_floats.data()), "Initialize");

        // Получаем информацию о модели
        std::cout << "Get model information..." << std::endl;
        NMDL_ModelInfo model_info = GetModelInformation(nmdl, 0);

        // Загрузка входного тензора
        std::cout << "Loading input tensor from file: " << FRAME_FILENAME << std::endl;
        std::vector<float> input = ReadFile<float>(FRAME_FILENAME);
        std::array<const float*, 1> inputs = { input.data() };

        // Выделение памяти для выходных данных
        std::cout << "Reserving output buffers..." << std::endl;
        std::vector<std::vector<float>> output_tensors(model_info.output_tensor_num);
        std::vector<float*> outputs(model_info.output_tensor_num);
        for(std::size_t i = 0; i < model_info.output_tensor_num; ++i) {
            std::size_t tensor_size = model_info.output_tensors[i].width *
                                      model_info.output_tensors[i].height *
                                      model_info.output_tensors[i].depth;
            output_tensors[i].resize(tensor_size);
            outputs[i] = output_tensors[i].data();
            std::cout << "Allocated " << tensor_size << " elements for output tensor " << i << std::endl;
        }

        // Запуск инференса для нескольких кадров
        std::cout << "Processing " << FRAMES << " frames..." << std::endl;
        double fps_total = 0.0;
        for(std::size_t frame = 0; frame < FRAMES; ++frame) {
            std::cout << "Processing frame " << frame << "..." << std::endl;
            Call(NMDL_Process(nmdl, 0, inputs.data()), "Process");
            fps_total += WaitForOutput(nmdl, 0, outputs.data());
        }
        std::cout << "Processed " << FRAMES << " frames." << std::endl;
        std::cout << "Average FPS: " << (fps_total / FRAMES) << std::endl;

        // Освобождение ресурсов
        NMDL_Release(nmdl);
        NMDL_Destroy(nmdl);
        std::cout << "Resources released. Exiting." << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if(nmdl) {
            NMDL_Release(nmdl);
            NMDL_Destroy(nmdl);
        }
        return 1;
    }
    return 0;
}
