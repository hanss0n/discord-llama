#include "LlamaModel.h"
#include <iostream>
#include <string>
#include <sstream>
#include <ctime>
#include <iomanip>

LlamaModel::LlamaModel(const std::string &model_path)
{

    std::string chatTranscript = generate_chat_transcript("test");
    llama_params = llama_context_default_params();

    ctx = llama_init_from_file(model_path.c_str(), llama_params);
    if (!ctx)
    {
        throw std::runtime_error("Failed to initialize the llama model from file: " + model_path);
    }

    generate_response(chatTranscript);
}

LlamaModel::~LlamaModel()
{
    llama_free(ctx);
}

std::string LlamaModel::generate_response(const std::string &input)
{

    std::vector<llama_token> tokens(llama_n_ctx(ctx));
    int token_count = llama_tokenize(ctx, input.c_str(), tokens.data(), tokens.size(), true);
    if (token_count < 0)
    {
        throw std::runtime_error("Failed to tokenize the input text.");
    }
    tokens.resize(token_count);

    int n_predict = 50; // Number of tokens to generate
    std::string output_str = "";

    for (int i = 0; i < n_predict; ++i)
    {
        int result = llama_eval(ctx, tokens.data(), token_count, 0, 10);
        if (result != 0)
        {
            throw std::runtime_error("Failed to run llama inference.");
        }

        llama_token top_token = llama_sample_top_p_top_k(ctx, tokens.data(), token_count, 40, 0.9f, 1.0f, 1.0f);
        const char *output_token_str = llama_token_to_str(ctx, top_token);

        output_str += std::string(output_token_str);

        std::cout << output_str << std::endl;

        // Update context with the generated token
        tokens.push_back(top_token);
        tokens.erase(tokens.begin());
    }

    return output_str;
}

std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    //assert(n >= 0);
    res.resize(n);

    return res;
}

std::string LlamaModel::prompt(const std::string &input)
{

    std::string prompt = input;
    prompt.insert(0, 1, ' ');
    std::vector<llama_token> tokens(llama_n_ctx(ctx));
    auto embeddings_input = ::llama_tokenize(ctx, prompt.c_str(), true);

    std::string result = "";
    
    int n_threads = 10;
    int n_keep = 200;    // idk what this is
    int n_remain = 2048; // change me
    int n_ctx = 2048;
    int n_consumed = 0;
    int n_past = 0;
    int repeat_last_n = 29; // idk?


    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    std::vector<llama_token> embeddings;

    while (n_remain != 0)
    {

        if (embeddings.size() > 0)
        {
            if (n_past + (int)embeddings.size() > n_ctx)
            {
                const int n_left = n_past - n_keep;
                n_past = n_keep;

                embeddings.insert(embeddings.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - embeddings.size(), last_n_tokens.end() - embeddings.size());
            }
            if (llama_eval(ctx, embeddings.data(), embeddings.size(), n_past, n_threads))
            {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                // throw std::runtime_error("%s : failed to eval\n"); fix this later
            }
        }

        n_past += embeddings.size();
        embeddings.clear();

        if ((int) embeddings_input.size() <= n_consumed)
        {
            const int32_t top_k = 0.7f;
            const float top_p = 0.8f;
            const float temp = 0.5f;
            const float repeat_penalty = 1.16f;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);

                // if (params.ignore_eos) {
                //   logits[llama_token_eos()] = 0; Is this needed?
                //}

                id = llama_sample_top_p_top_k(ctx,
                                              last_n_tokens.data() + n_ctx - repeat_last_n,
                                              repeat_last_n, top_k, top_p, temp, repeat_penalty);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }
            embeddings.push_back(id);
            --n_remain;
        }
        else
        {

            while ((int)embeddings_input.size() > n_consumed)
            {
                embeddings.push_back(embeddings_input[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embeddings_input[n_consumed]);
                ++n_consumed;
                if ((int)embeddings.size() >= params.n_batch)
                {
                    break;
                }
            }
        }
    }
    for (auto id : embeddings) {
        result += llama_token_to_str(ctx, id);
    }

    return result;
}

std::string LlamaModel::generate_chat_transcript(const std::string &user_name, const std::string &ai_name)
{
    std::ostringstream output;

    output << "Text transcript of a never ending dialog, where " << user_name << " interacts with an AI assistant named " << ai_name << ".\n"
           << ai_name << " is helpful, kind, honest, friendly, good at writing and never fails to answer " << user_name << "’s requests immediately and with details and precision.\n"
           << "There are no annotations like (30 seconds passed...) or (to himself), just what " << user_name << " and " << ai_name << " say aloud to each other.\n"
           << "The dialog lasts for years, the entirety of it is shared below. It's 10000 pages long.\n"
           << "The transcript only includes text, it does not include markup like HTML and Markdown.\n"
           << user_name << ": Hello, " << ai_name << "!\n"
           << ai_name << ": Hello " << user_name << "! How may I help you today?\n"
           << user_name << ": What time is it?\n"
           << ai_name << ": It is $(date +%H:%M).\n"
           << user_name << ": What year is it?\n"
           << ai_name << ": We are in $(date +%Y).\n"
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
           << user_name << ": Name a color.\n"
           << ai_name << ": Blue\n";

    return output.str();
}
