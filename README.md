# Discord Llama
Simple Discord bot that uses Meta's Llama language model.

## D++ Template
This repo uses the CMake template for a simple [D++](https://dpp.dev) bot. This template assumes that D++ is already installed.

## Prerequisites 
* Place llama 13B weights under llama.cpp/models and run quantization, instructions can be found here: https://github.com/ggerganov/llama.cpp
* Compile llama.cpp as a shared library with CMake

## Compilation

    mkdir build
    cd build
    cmake ..
    make -j

If DPP is installed in a different location you can specify the root directory to look in while running cmake 

    cmake .. -DDPP_ROOT_DIR=<your-path>

## Running the discord llama bot

Create a config.json in the directory above the build directory:

```json
{ "token": "your bot token here" }
```

Start the bot:

    cd build
    ./discord_llama



