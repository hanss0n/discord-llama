#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include <string>
#include <vector>
#include <stdexcept>
#include "llama.h"
#include "../llama.cpp/examples/common.h"

class LlamaModel {
public:
    LlamaModel(const std::string& model_path);
    ~LlamaModel();
    std::string generate_response(const std::string& input);
    std::string prompt(const std::string& input);

private:
    struct llama_context *ctx;
    struct gpt_params params;
    struct llama_context_params llama_params;
    std::string generate_chat_transcript(const std::string& USER_NAME, const std::string& AI_NAME = "Cheems");
};

#endif // LLAMA_MODEL_H