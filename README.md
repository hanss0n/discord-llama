# Discord Llama Bot

The Discord Llama Bot is a simple Discord bot that utilizes Meta's Llama language model to interact with users. This repository is developed using [llama.cpp](https://github.com/ggerganov/llama.cpp) and [D++ (DPP)](https://github.com/brainboxdotcc/DPP).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Compilation](#compilation)
- [Configuration](#configuration)
- [Running the Bot](#running-the-bot)
- [Contribution Guidelines](#contribution-guidelines)

## Prerequisites

Before you begin, please make sure you have completed the following steps:

1. Place the Llama 13B weights under `llama.cpp/models` and run quantization. You can find the instructions [here](https://github.com/ggerganov/llama.cpp).
2. Compile `llama.cpp` as a shared library with CMake.

## Compilation

To compile the Discord Llama Bot, follow these steps:

```bash
mkdir build
cd build
cmake ..
make -j
```

If DPP is installed in a different location you can specify the root directory to look in while running cmake 

    cmake .. -DDPP_ROOT_DIR=<your-path>

## Configuration

Create a config.json in the directory above the build directory:

```json
{ "token": "your bot token here" }
```

## Running the Bot:
```bash
cd build
./discord_llama
```

## Contribution Guidelines

We welcome contributions to the Discord Llama Bot project! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a branch with a clear and descriptive name related to the changes you will be making.
3. Make your changes in the branch.
4. Ensure that your code is properly formatted and adheres to the coding style of the project.
5. Test your changes to ensure that they are functioning correctly and do not introduce new bugs.
6. Create a pull request with a clear and descriptive title and description that outlines the changes you've made and their purpose.

Please note that any contributions you make to this project will be under the same license as the original project. By submitting a pull request, you agree to these terms.

Thank you for considering contributing to the Discord Llama Bot project! We look forward to collaborating with you.




