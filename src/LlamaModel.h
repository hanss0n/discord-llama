#ifndef LLAMA_MODEL_H
#define LLAMA_MODEL_H

#include <string>
#include <vector>
#include <stdexcept>
#include "llama.h"
#include <filesystem>
#include <thread>

struct gpt_params {
    int32_t seed = -1;
    int32_t n_threads = std::min(4, static_cast<int32_t>(std::thread::hardware_concurrency()));
    int32_t n_predict = 256;
    int32_t repeat_last_n = 256;
    int32_t n_parts = -1;
    int32_t n_ctx = 10000;
    int32_t n_batch = 1024;
    int32_t n_keep = 0;

    int32_t top_k = 40;
    float top_p = 0.95f;
    float temp = 0.80f;
    float repeat_penalty = 1.10f;

    std::filesystem::path model = std::filesystem::path("models") / "13B" / "ggml-model-q4_0.bin";
    std::string prompt;
    std::string input_prefix;

    std::vector<std::string> antiprompt;

    bool memory_f16 = true;
    bool random_prompt = false;
    bool use_color = false;
    bool interactive = false;

    bool embedding = false;
    bool interactive_start = false;

    bool instruct = false;
    bool ignore_eos = false;
    bool perplexity = false;
    bool use_mlock = false;
    bool mem_test = false;
    bool verbose_prompt = false;
};

class LlamaModel
{
public:
    LlamaModel(const std::filesystem::path &model_path, const std::string &user_name, const std::string &ai_name);
    ~LlamaModel();
    std::string prompt(const std::string &input);

private:
    struct llama_context *ctx;
    struct llama_context_params llama_params;
    std::string generate_chat_transcript(const std::string &user_name, const std::string &ai_name);
    void consume_tokens(std::vector<llama_token> &embeddings, const std::vector<llama_token> &embeddings_input, std::vector<llama_token> &last_n_tokens, int &n_consumed);
    std::vector<llama_token> process_embeddings(std::vector<llama_token> &embeddings, std::vector<llama_token> &last_n_tokens, int &n_past);
    struct gpt_params params;
    bool is_antiprompt_detected(const std::vector<llama_token> &last_n_tokens, const std::vector<std::string> &antiprompts, llama_context *ctx);
    std::string remove_prefix_and_suffix(std::string str, const std::string &prefix, const std::string &suffix);
    std::string ai_name;
    std::string user_name;
    std::string to_lower(const std::string &input,
                         std::string::const_iterator start = std::string::const_iterator(),
                         std::string::const_iterator end = std::string::const_iterator());

    std::vector<llama_token> tokens;
    int n_past;
    std::vector<llama_token> last_n_tokens;
    std::vector<llama_token> embeddings;
};

#endif // LLAMA_MODEL_H