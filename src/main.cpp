#include <iostream>
#include "LlamaModel.h"

int main() {
    try {
        // Replace "path/to/model/file" with the actual path to your llama model file
        LlamaModel model("path/to/model/file");

        std::string input_text;
        while (true) {
            std::cout << "Enter your message (type 'exit' to quit): ";
            std::getline(std::cin, input_text);

            if (input_text == "exit") {
                break;
            }

            std::string response = model.generate_response(input_text);
            std::cout << "LlamaModel: " << response << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
