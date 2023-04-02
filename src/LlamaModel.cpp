#include "LlamaModel.h"

LlamaModel::LlamaModel(const std::string& model_path) {
    llama_context_params params = llama_context_default_params();
    ctx = llama_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        throw std::runtime_error("Failed to initialize the llama model from file: " + model_path);
    }
}

LlamaModel::~LlamaModel() {
    llama_free(ctx);
}

std::string LlamaModel::generate_response(const std::string& input) {
    std::vector<llama_token> tokens(llama_n_ctx(ctx));
    int token_count = llama_tokenize(ctx, input.c_str(), tokens.data(), tokens.size(), true);
    if (token_count < 0) {
        throw std::runtime_error("Failed to tokenize the input text.");
    }
    tokens.resize(token_count);

    int result = llama_eval(ctx, tokens.data(), token_count, 0, 1);
    if (result != 0) {
        throw std::runtime_error("Failed to run llama inference.");
    }

    llama_token top_token = llama_sample_top_p_top_k(ctx, tokens.data(), token_count, 1, 0.0f, 1.0f, 1.0f);
    const char *output_token_str = llama_token_to_str(ctx, top_token);

    return std::string(output_token_str);
}
