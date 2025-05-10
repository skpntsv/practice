#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint> 

#include "nmdl.h"
#include "nmdl_compiler.h"
#include "nmdl_image_converter.h"

//#define _USE_DARKNET_

namespace {

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
	};
}

template <typename T>
std::vector<T> ReadFile(const std::string &filename) {
	std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
	if(!ifs.is_open()) {
		throw std::runtime_error("Unable to open input file: " + filename);
	}
	auto fsize = static_cast<std::size_t>(ifs.tellg());
	ifs.seekg(0);
	std::vector<T> data(fsize / sizeof(T));
	ifs.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(T));
	return data;
}

void ShowNMDLVersion() {
	std::uint32_t major = 0;
	std::uint32_t minor = 0;
	std::uint32_t patch = 0;
	Call(NMDL_GetLibVersion(&major, &minor, &patch), "GetLibVersion");
	std::cout << "Lib version: " << major << "." << minor 
			<< "." << patch << std::endl;
}

void CheckBoard(std::uint32_t required_board_type) {
	std::uint32_t boards = 0;
	std::uint32_t board_number = -1;
	Call(NMDL_GetBoardCount(required_board_type, &boards), "GetBoardCount");
	std::cout << "Detected boards: " << boards << std::endl;
	if(!boards) {
		throw std::runtime_error("Board not found");
	}
}

#ifdef _USE_DARKNET_
std::vector<float> CompileModel(const std::string &config_filename, 
		const std::string &weights_filename, 
		std::uint32_t board_type,
		bool is_multi_unit) {
	float *nm_model = nullptr;
	std::uint32_t nm_model_floats = 0u;
	auto config = ReadFile<char>(config_filename);
	auto weights = ReadFile<char>(weights_filename);
	Call(NMDL_COMPILER_CompileDarkNet(is_multi_unit, board_type, 
			config.data(), config.size(), weights.data(), weights.size(),
			&nm_model, &nm_model_floats), "CompileONNX");
	std::vector<float> result(nm_model, nm_model + nm_model_floats);
	NMDL_COMPILER_FreeModel(board_type, nm_model);
	return result;
}
#else
std::vector<float> CompileModel(const std::string &model_filename, std::uint32_t board_type,
		bool is_multi_unit) {
	float *nm_model = nullptr;
	std::uint32_t nm_model_floats = 0u;
	auto model = ReadFile<char>(model_filename);
	Call(NMDL_COMPILER_CompileONNX(is_multi_unit, board_type, model.data(),
			static_cast<std::uint32_t>(model.size()), 
			&nm_model, &nm_model_floats), "CompileONNX");
	std::vector<float> result(nm_model, nm_model + nm_model_floats);
	NMDL_COMPILER_FreeModel(board_type, nm_model);
	return result;
}
#endif

NMDL_ModelInfo GetModelInformation(NMDL_HANDLE nmdl, std::uint32_t unit_num) {
	NMDL_ModelInfo model_info;
	Call(NMDL_GetModelInfo(nmdl, unit_num, &model_info), "GetModelInfo");
	std::cout << "Input tensor number: " << model_info.input_tensor_num << std::endl;
	for(std::size_t i = 0; i < model_info.input_tensor_num; ++i) {
		std::cout << "Input tensor " << i << ": W-" <<
			model_info.input_tensors[i].width << ", H-" <<
			model_info.input_tensors[i].height << ", D-" <<
			model_info.input_tensors[i].depth <<
			std::endl;
	}
	std::cout << "Output tensor number: " << model_info.output_tensor_num << std::endl;
	for(std::size_t i = 0; i < model_info.output_tensor_num; ++i) {
		std::cout << "Output tensor " << i << ": W-" <<
			model_info.output_tensors[i].width << ", H-" <<
			model_info.output_tensors[i].height << ", D-" <<
			model_info.output_tensors[i].depth <<
			std::endl;
	}
	return model_info;
}

std::vector<float> PrepareInput(const std::string &filename, std::uint32_t width, 
		std::uint32_t height, std::uint32_t board_type, 
		std::uint32_t color_format, const float rgb_divider[3], 
		const float rgb_adder[3]) {
	auto bmp_frame = ReadFile<char>(filename);
	std::vector<float> input(NMDL_IMAGE_CONVERTER_RequiredSize(
			width, height, color_format, board_type));
	if(NMDL_IMAGE_CONVERTER_Convert(bmp_frame.data(), input.data(), 
			static_cast<std::uint32_t>(bmp_frame.size()), width, height,
			color_format, rgb_divider, rgb_adder, board_type)) {
		throw std::runtime_error("Image conversion error");
	}
	return input;
}

void WaitForOutput(NMDL_HANDLE nmdl, std::uint32_t unit_num, float *outputs[]) {
	std::uint32_t status = NMDL_PROCESS_FRAME_STATUS_INCOMPLETE;
	while(status == NMDL_PROCESS_FRAME_STATUS_INCOMPLETE) {
		NMDL_GetStatus(nmdl, unit_num, &status);
	};
	double fps;

	Call(NMDL_GetOutput(nmdl, unit_num, outputs, &fps), "GetOutput");
	std::cout << "First 20 result values:" << std::endl;
	for(std::size_t i = 0; i < 20; ++i) {
		std::cout << outputs[0][i] << std::endl;
	}
	std::cout << "FPS:" << fps << std::endl;
}

}

int main() {
	// const std::uint32_t BOARD_TYPE = NMDL_BOARD_TYPE_SIMULATOR;
	//const uint32_t BOARD_TYPE = NMDL_BOARD_TYPE_MC12705;
	const uint32_t BOARD_TYPE = NMDL_BOARD_TYPE_NMCARD;
	//const uint32_t BOARD_TYPE = NMDL_BOARD_TYPE_NMMEZZO;
	//const uint32_t BOARD_TYPE = NMDL_BOARD_TYPE_NMQUAD;
	//const uint32_t BOARD_TYPE = NMDL_BOARD_TYPE_MC12101;
	//const uint32_t BOARD_TYPE = NMDL_BOARD_TYPE_NMSTICK;
	//const uint32_t BOARD_TYPE = NMDL_BOARD_TYPE_NMEMBED;
	const std::uint32_t COMPILER_BOARD_TYPE =
		(BOARD_TYPE == NMDL_BOARD_TYPE_MC12101 ||
		BOARD_TYPE == NMDL_BOARD_TYPE_NMSTICK) ?
		NMDL_COMPILER_BOARD_TYPE_MC12101 :
		NMDL_COMPILER_BOARD_TYPE_MC12705;
	const std::uint32_t IMAGE_CONVERTER_BOARD_TYPE =
		(BOARD_TYPE == NMDL_BOARD_TYPE_MC12101 ||
		BOARD_TYPE == NMDL_BOARD_TYPE_NMSTICK) ?
		NMDL_IMAGE_CONVERTER_BOARD_TYPE_MC12101 :
		NMDL_IMAGE_CONVERTER_BOARD_TYPE_MC12705;
#ifdef _USE_DARKNET_ 
	const std::string DARKNET_CONFIG_FILENAME = 
			"../nmdl_ref_data/yolo_v3_tiny_coco/model.cfg";
	const std::string DARKNET_WEIGHTS_FILENAME =
			"../nmdl_ref_data/yolo_v3_tiny_coco/model.weights";
	const std::string BMP_FRAME_FILENAME = 
			"../nmdl_ref_data/yolo_v3_tiny_coco/frame.bmp";
	const NMDL_IMAGE_CONVERTER_COLOR_FORMAT IMAGE_CONVERTER_COLOR_FORMAT =
		NMDL_IMAGE_CONVERTER_COLOR_FORMAT_RGB;
	const float NM_FRAME_RGB_DIVIDER[3] = {255.0f, 255.0f, 255.0f};
	const float NM_FRAME_RGB_ADDER[3] = {0.0f, 0.0f, 0.0f};
#else
	const std::string ONNX_MODEL_FILENAME = 
			"../model/model.onnx";
	const std::string BMP_FRAME_FILENAME =
			"../data/sample.bmp";
	const NMDL_IMAGE_CONVERTER_COLOR_FORMAT IMAGE_CONVERTER_COLOR_FORMAT =
		NMDL_IMAGE_CONVERTER_COLOR_FORMAT_INTENSITY;
	const float NM_FRAME_RGB_DIVIDER[3] = {255.0f, 255.0f, 255.0f};
	const float NM_FRAME_RGB_ADDER[3] = {0.0f, 0.0f, 0.0f};
#endif
	const std::size_t BATCHES = 4;
	const std::size_t FRAMES = 1;

	NMDL_HANDLE nmdl = 0;

	try {
		std::cout << "Query library version..." << std::endl;
		ShowNMDLVersion();

		std::cout << "Board detection... " << std::endl;
		CheckBoard(BOARD_TYPE);
			
		std::cout << "NMDL initialization... " << std::endl;
		Call(NMDL_Create(&nmdl), "Create");
		
		std::cout << "Use multi unit... " << std::endl;

		std::cout << "Compile model... " << std::endl;
#ifdef _USE_DARKNET_
		auto model = CompileModel(DARKNET_CONFIG_FILENAME, DARKNET_WEIGHTS_FILENAME, 
				COMPILER_BOARD_TYPE, true);
#else
		auto model = CompileModel(ONNX_MODEL_FILENAME, COMPILER_BOARD_TYPE, false);
#endif

		std::array<const float*, NMDL_MAX_UNITS> models = {model.data()};
		std::array<std::uint32_t, NMDL_MAX_UNITS> model_floats = 
				{static_cast<std::uint32_t>(model.size())};
		Call(NMDL_Initialize(nmdl, BOARD_TYPE, 0, 0, models.data(), 
				model_floats.data()), "Initialize");

		std::cout << "Get model information... " << std::endl;
		auto model_info = GetModelInformation(nmdl, 0);

		std::cout << "Prepare inputs... " << std::endl;

		auto input = PrepareInput(BMP_FRAME_FILENAME, model_info.input_tensors[0].width,
				model_info.input_tensors[0].height, IMAGE_CONVERTER_BOARD_TYPE,
				IMAGE_CONVERTER_COLOR_FORMAT, NM_FRAME_RGB_DIVIDER, NM_FRAME_RGB_ADDER);
		std::array<const float*, 1> inputs = {input.data()};
		// auto input = ReadFile<float>("../data/sample.raw");
		// std::array<const float*,1> inputs = { input.data() };

		std::cout << "Reserve outputs... " << std::endl;
		std::vector<std::vector<float>> output_tensors(model_info.output_tensor_num);
		std::vector<float*> outputs(model_info.output_tensor_num);
		for(std::size_t i = 0; i < model_info.output_tensor_num; ++i) {
			output_tensors[i].resize(static_cast<std::size_t>(
					model_info.output_tensors[i].width) *
					model_info.output_tensors[i].height * 
					model_info.output_tensors[i].depth);
			outputs[i] = output_tensors[i].data();
		}

		std::cout << "Process inputs... " << std::endl;
		for(std::size_t i = 0; i < FRAMES; ++i) {
			Call(NMDL_Process(nmdl, 0, inputs.data()), "Process");
			WaitForOutput(nmdl, 0, outputs.data());
		}
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
	NMDL_Release(nmdl);
	NMDL_Destroy(nmdl);

	return 0;
}
