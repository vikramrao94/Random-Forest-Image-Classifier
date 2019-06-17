Place images to be trained in "images" folder.<br />
**Dependencies:**<br />
1.OpenCV<br />
2.https://github.com/nlohmann/json<br />
**Build:**<br />
g++ train.cpp -o trainer `pkg-config --cflags --libs opencv`<br />
g++ predict.cpp -o predict `pkg-config --cflags --libs opencv`<br />
**Run:**<br />
./trainer<br />
./predict<br />
