#include "LlamaModel.h"
#include <iostream>
#include <string>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <algorithm>

LlamaModel::LlamaModel(const std::string &model_path)
{

    user_name = "### Discord user";
    ai_name = "### Cheems";

    std::string chat_transcript = generate_chat_transcript(user_name, ai_name);
    params.antiprompt.push_back(user_name);
    llama_params = llama_context_default_params();
    llama_params.n_ctx = params.n_ctx;
    llama_params.n_parts = params.n_parts;
    llama_params.seed = params.seed;
    llama_params.f16_kv = params.memory_f16;

    ctx = llama_init_from_file(model_path.c_str(), llama_params);
    if (!ctx)
    {
        throw std::runtime_error("Failed to initialize the llama model from file: " + model_path);
    }

    prompt(chat_transcript);
}

LlamaModel::~LlamaModel()
{
    llama_free(ctx);
}

std::vector<llama_token> llama_tokenize(struct llama_context *ctx, const std::string &text, bool add_bos)
{
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    // assert(n >= 0);
    res.resize(n);

    return res;
}

std::vector<llama_token> LlamaModel::process_embeddings(std::vector<llama_token> &embeddings, std::vector<llama_token> &last_n_tokens, int &n_past)
{
    if (n_past + static_cast<int>(embeddings.size()) > params.n_ctx)
    {
        const int n_left = n_past - params.n_keep;
        n_past = params.n_keep;
        embeddings.insert(embeddings.begin(), last_n_tokens.begin() + params.n_ctx - n_left / 2 - embeddings.size(), last_n_tokens.end() - embeddings.size());
    }
    if (llama_eval(ctx, embeddings.data(), embeddings.size(), n_past, params.n_threads))
        throw std::runtime_error("Failed to evaluate Llama model");

    n_past += embeddings.size();
    embeddings.clear();

    return embeddings;
}

void LlamaModel::consume_tokens(std::vector<llama_token> &embeddings, const std::vector<llama_token> &embeddings_input, std::vector<llama_token> &last_n_tokens, int &n_consumed)
{
    while (static_cast<int>(embeddings_input.size()) > n_consumed)
    {
        llama_token current_token = embeddings_input[n_consumed];
        embeddings.push_back(current_token);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(current_token);
        ++n_consumed;

        if (static_cast<int>(embeddings.size()) >= params.n_batch)
            break;
    }
}

bool LlamaModel::is_antiprompt_detected(const std::vector<llama_token> &last_n_tokens, const std::vector<std::string> &antiprompts, llama_context *ctx)
{
    std::string last_output;
    for (auto id : last_n_tokens)
        last_output += llama_token_to_str(ctx, id);

    for (const std::string &antiprompt : antiprompts)
    {
        size_t extra_padding = 2;
        size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                                      ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                                      : 0;
        if (last_output.find(antiprompt.c_str(), search_start_pos) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

std::string LlamaModel::remove_prefix_and_suffix(std::string str, const std::string &prefix, const std::string &suffix) {
    // Remove prefix from the front
    if (str.find(prefix) == 0) {
        str.erase(0, prefix.length() + 2);
    }

    // Remove suffix from the back
    if (str.rfind(suffix) == str.length() - suffix.length()) {
        str.erase(str.length() - suffix.length(), suffix.length());
    }

    return str;
}

std::string LlamaModel::prompt(const std::string &input)
{
    std::string prompt = " " + input;
    std::vector<llama_token> tokens(llama_n_ctx(ctx));
    auto embeddings_input = ::llama_tokenize(ctx, prompt.c_str(), true);

    std::string result;
    int n_remain = params.n_predict, n_consumed = 0, n_past = 0;

    std::vector<llama_token> last_n_tokens(params.n_ctx, 0);
    std::vector<llama_token> embeddings, result_vector;
    bool is_antiprompt = false;

    while (n_remain != 0 && !is_antiprompt)
    {
        if (!embeddings.empty())
            embeddings = process_embeddings(embeddings, last_n_tokens, n_past);

        if (static_cast<int>(embeddings_input.size()) <= n_consumed)
        {
            llama_token id = llama_sample_top_p_top_k(ctx, last_n_tokens.data() + params.n_ctx - params.repeat_last_n, params.repeat_last_n, params.top_k, params.top_p, params.temp, params.repeat_penalty);
            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(id);

            embeddings.push_back(id);
            result += llama_token_to_str(ctx, id);
            std::cout << result << std::endl;
            --n_remain;
        }
        else
            consume_tokens(embeddings, embeddings_input, last_n_tokens, n_consumed);

        result_vector.insert(result_vector.end(), embeddings.begin(), embeddings.end());

        if (static_cast<int>(embeddings_input.size()) <= n_consumed)
        {
            is_antiprompt = is_antiprompt_detected(last_n_tokens, params.antiprompt, ctx);
        }
    }

    std::cout << result.length() << std::endl;
    std::cout << remove_prefix_and_suffix(result, ai_name, user_name) << std::endl;

    return remove_prefix_and_suffix(result, ai_name, user_name);
}

std::string LlamaModel::generate_chat_transcript(const std::string &user_name, const std::string &ai_name)
{
    std::ostringstream output;
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm *current_time = std::localtime(&now_c);

    char time_buffer[6];
    std::strftime(time_buffer, sizeof(time_buffer), "%H:%M", current_time);

    char year_buffer[5];
    std::strftime(year_buffer, sizeof(year_buffer), "%Y", current_time);

    std::string current_time_str(time_buffer);
    std::string current_year_str(year_buffer);

    output << "Text transcript of a never ending dialog, where " << user_name << " interacts with an AI assistant named " << ai_name << ".\n"
           << ai_name << " is helpful, kind, honest, friendly, good at writing and never fails to answer " << user_name << "’s requests immediately and with details and precision.\n"
           << "There are no annotations like (30 seconds passed...) or (to himself), just what " << user_name << " and " << ai_name << " say aloud to each other.\n"
           << "The dialog lasts for years, the entirety of it is shared below. It's 10000 pages long.\n"
           << "The transcript only includes text, it does not include markup like HTML and Markdown.\n"
           << user_name << ": Hello, " << ai_name << "!\n"
           << ai_name << ": Hello " << user_name << "! How may I help you today?\n"
           << user_name << ": What time is it?\n"
           << ai_name << ": It is " << current_time_str << ".\n"
           << user_name << ": What year is it?\n"
           << ai_name << ": We are in " << current_year_str << ".\n"
           << user_name << ": Please tell me the largest city in Europe.\n"
           << ai_name << ": The largest city in Europe is Moscow, the capital of Russia.\n"
           << user_name << ": What can you tell me about Moscow?\n"
           << ai_name << ": Moscow, on the Moskva River in western Russia, is the nation’s cosmopolitan capital. In its historic core is the Kremlin, a complex that’s home to the president and tsarist treasures in the Armoury. Outside its walls is Red Square, Russia’s symbolic center.\n"
           << user_name << ": What is a cat?\n"
           << ai_name << ": A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.\n"
           << user_name << ": How do I pass command line arguments to a Node.js program?\n"
           << ai_name << ": The arguments are stored in process.argv.\n"
           << "    argv[0] is the path to the Node. js executable.\n"
           << "    argv[1] is the path to the script file.\n"
           << "    argv[2] is the first argument passed to the script.\n"
           << "    argv[3] is the second argument passed to the script and so on.\n"
           << user_name << ": Name one color.\n";

    return output.str();
}
