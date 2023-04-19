#include <iostream>
#include "LlamaModel.h"
#include <deps/deps.h>
#include <sstream>
#include <filesystem>

std::string append_dot_if_needed(const std::string &input) {
    if (input.empty()) {
        return input;
    }

    char last_char = input.back();
    if (last_char != '.' && last_char != '?' && last_char != '!') {
        return input + '.';
    }

    return input;
}

int main()
{

    try
    {
        std::string user_name = "User@1243";
        std::string ai_name = "BDU@8738";
        std::filesystem::path llama_src_location = std::filesystem::current_path().append("..").append("llama.cpp");
        LlamaModel model(llama_src_location, user_name, ai_name);

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

        bot.on_ready([&command_handler, &model, &bot, &user_name](const dpp::ready_t &event)
                     {
                         command_handler.add_command(
                             /* Command name */
                             "llama",

                             /* Parameters */
                             {
                                 {"text", dpp::param_info(dpp::pt_string, true, "LLaMa prompt")}},

                             /* Command handler */
                             [&command_handler, &model, &user_name](const std::string &command, const dpp::parameter_list_t &parameters, dpp::command_source src)
                             {
                                 std::string got_param;
                                 if (!parameters.empty())
                                 {
                                     got_param = std::get<std::string>(parameters[0].second);
                                 }

                                dpp::command_completion_event_t original_callback = [](const dpp::confirmation_callback_t& cc) {};

                                command_handler.thinking(src);
                                std::string generated_response = model.prompt(user_name + ": " + append_dot_if_needed(got_param) + "\n");
                                command_handler.owner->interaction_followup_edit_original(src.command_token, dpp::message("> " + got_param + "\n\n" + generated_response));
                             },

                             /* Command description */
                             "Prompt LLaMa language model"
                             );

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
