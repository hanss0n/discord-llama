#include "LlamaModel.h"
#include <iostream>

LlamaModel::LlamaModel(const std::string &model_path)
{
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 1024;
    params.seed = 42;
    params.n_parts = -1;
    params.f16_kv = false;
    params.logits_all = false;
    params.vocab_only = false;
    params.use_mlock = false;
    params.embedding = false;



    ctx = llama_init_from_file(model_path.c_str(), params);
    if (!ctx)
    {
        throw std::runtime_error("Failed to initialize the llama model from file: " + model_path);
    }
}

LlamaModel::~LlamaModel()
{
    llama_free(ctx);
}

std::string LlamaModel::generate_response(const std::string& input) {

    std::vector<llama_token> tokens(llama_n_ctx(ctx));
    int token_count = llama_tokenize(ctx, input.c_str(), tokens.data(), tokens.size(), true);
    if (token_count < 0) {
        throw std::runtime_error("Failed to tokenize the input text.");
    }
    tokens.resize(token_count);

    int n_predict = 50; // Number of tokens to generate
    std::string output_str;
    
    for (int i = 0; i < n_predict; ++i) {
        int result = llama_eval(ctx, tokens.data(), token_count, 0, 8);
        if (result != 0) {
            throw std::runtime_error("Failed to run llama inference.");
        }

        llama_token top_token = llama_sample_top_p_top_k(ctx, tokens.data(), token_count, 40, 0.9f, 1.0f, 1.0f);
        const char *output_token_str = llama_token_to_str(ctx, top_token);

        output_str += std::string(output_token_str);
        std::cout << output_str << std::endl;
        
        // Update context with the generated token
        tokens.push_back(top_token);
        token_count++;
    }

    return output_str;
}
