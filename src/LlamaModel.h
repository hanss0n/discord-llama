#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include <string>
#include <vector>
#include <stdexcept>
#include "llama.h"
#include <thread>

class LlamaModel
{
public:
    LlamaModel(const std::string &model_path);
    ~LlamaModel();
    std::string generate_response(const std::string &input);
    std::string prompt(const std::string &input);

private:
    struct llama_context *ctx;
    struct llama_context_params llama_params;
    std::string generate_chat_transcript(const std::string &user_name, const std::string &ai_name);
    void consume_tokens(std::vector<llama_token> &embeddings, const std::vector<llama_token> &embeddings_input, std::vector<llama_token> &last_n_tokens, int &n_consumed);
    std::vector<llama_token> process_embeddings(std::vector<llama_token> &embeddings, std::vector<llama_token> &last_n_tokens, int &n_past);
    struct gpt_params
    {
        int32_t seed = -1; // RNG seed
        int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
        int32_t n_predict = 512;    // new tokens to predict
        int32_t repeat_last_n = 256; // last n tokens to penalize
        int32_t n_parts = -1;       // amount of model parts (-1 = determine from model dimensions)
        int32_t n_ctx = 2048;        // context size
        int32_t n_batch = 1024;        // batch size for prompt processing
        int32_t n_keep = 0;         // number of tokens to keep from initial prompt

        // sampling parameters
        int32_t top_k = 40;
        float top_p = 0.95f;
        float temp = 0.80f;
        float repeat_penalty = 1.10f;

        std::string model = "models/lamma-7B/ggml-model.bin"; // model path
        std::string prompt = "";
        std::string input_prefix = ""; // string to prefix user inputs with

        std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted

        bool memory_f16 = true;     // use f16 instead of f32 for memory kv
        bool random_prompt = false; // do not randomize prompt if none provided
        bool use_color = false;     // use color to distinguish generations and inputs
        bool interactive = false;   // interactive mode

        bool embedding = false;         // get only sentence embedding
        bool interactive_start = false; // wait for user input immediately

        bool instruct = false;       // instruction mode (used for Alpaca models)
        bool ignore_eos = false;     // do not stop generating after eos
        bool perplexity = false;     // compute perplexity over the prompt
        bool use_mlock = false;      // use mlock to keep model in memory
        bool mem_test = false;       // compute maximum memory usage
        bool verbose_prompt = false; // print prompt tokens before generation
    };
    struct gpt_params params;
    bool is_antiprompt_detected(const std::vector<llama_token> &last_n_tokens, const std::vector<std::string> &antiprompts, llama_context *ctx);
    std::string remove_prefix_and_suffix(std::string str, const std::string &prefix, const std::string &suffix);
    std::string ai_name;
    std::string user_name;
};

#endif // LLAMA_MODEL_H