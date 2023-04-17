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
        std::cout << model.prompt("John: Who is Harry Potter?") << std::endl;
    }
    catch (const std::runtime_error &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
