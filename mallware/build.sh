#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ..
cmake --build . --config Release
cp NMCardMini_Inference /opt/nmdl/bin
cd ..
