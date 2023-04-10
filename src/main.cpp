#include <iostream>
#include "LlamaModel.h"
#include <templatebot/templatebot.h>
#include <sstream>

int main()
{

    try
    {
        // Replace "path/to/model/file" with the actual path to your llama model file
        LlamaModel model("/home/bjorn/Projects/discord-llama/llama.cpp/models/13B/ggml-model-q4_0.bin");

        json configdocument;
        std::ifstream configfile("../config.json");
        configfile >> configdocument;

        /* Setup the bot */
        dpp::cluster bot(configdocument["token"]);

        bot.on_log(dpp::utility::cout_logger());

        /* Create command handler, and specify prefixes */
        dpp::commandhandler command_handler(&bot);
        /* Specifying a prefix of "/" tells the command handler it should also expect slash commands */
        command_handler.add_prefix(".").add_prefix("/");

    

        bot.on_ready([&command_handler, &model, &bot](const dpp::ready_t &event)
                     {
                         command_handler.add_command(
                             /* Command name */
                             "llama",

                             /* Parameters */
                             {
                                 {"text", dpp::param_info(dpp::pt_string, true, "LLaMa prompt")}},

                             /* Command handler */
                             [&command_handler, &model, &bot](const std::string &command, const dpp::parameter_list_t &parameters, dpp::command_source src)
                             {
                                 std::string got_param;
                                 if (!parameters.empty())
                                 {
                                     got_param = std::get<std::string>(parameters[0].second);
                                 }
                                command_handler.reply(dpp::message("  "), src);
                                std::string response = model.generate_response(got_param);
                                bot.message_create(dpp::message(src.channel_id, response));
                                
                             },

                             /* Command description */
                             "Prompt LLaMa language model");

                         /* NOTE: We must call this to ensure slash commands are registered.
                          * This does a bulk register, which will replace other commands
                          * that are registered already!
                          */
                         command_handler.register_commands(); });

        bot.start(dpp::st_wait);
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
