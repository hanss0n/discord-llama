#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include <string>
#include <vector>
#include <stdexcept>
#include "llama.h"

class LlamaModel {
public:
    LlamaModel(const std::string& model_path);
    ~LlamaModel();
    std::string generate_response(const std::string& input);

private:
    struct llama_context *ctx;
};

#endif // LLAMA_MODEL_H