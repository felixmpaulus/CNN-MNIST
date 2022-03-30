"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const cnn_1 = require("./cnn");
main();
function main() {
    const weightFile = null; // 'squares5'
    const options = { activation: 'ReLU', weightOptions: { weightFile, lowerLimit: -0.25, higherLimit: 0.25 } };
    const MNISTCNN = new cnn_1.CNN(2, [2], 1, options);
    console.log('initial weights: ');
    console.log(MNISTCNN.beautifyWeights(MNISTCNN.weights));
    const data4Squares = [
        { input: [0, 0, 1, 1], label: [1] },
        { input: [1, 1, 0, 0], label: [1] },
        { input: [1, 0, 1, 0], label: [1] },
        { input: [0, 1, 0, 1], label: [1] },
        { input: [1, 0, 0, 1], label: [0] },
        { input: [0, 1, 1, 0], label: [0] }
    ];
    const dataXOR = [
        { input: [1, 1], label: [0] },
        { input: [0, 1], label: [1] },
        { input: [1, 0], label: [1] },
        { input: [0, 0], label: [0] },
    ];
    train(MNISTCNN, dataXOR, 250, 2, weightFile);
    detect(MNISTCNN, dataXOR);
}
// 3 error -> 0
// 2 error -> ?
// 1 error -> {0, 0.5}
// XOR
// ReLU, l=0.3, w=[-0.25, 0.25]
// 1111 ...
// 3: 13x (43%)
// 2: 4x  (14%)
// 1: 13x (43%)
// ReLU, l=0.3, w=[-0.5, 0.5]
// 1111 ...
// 3: 13x (43%)
// 2: 4x  (14%)
// 1: 13x (43%)
// squares
// ReLU, l=0.3, w=[0,1]
// 1331332321 3311311111 2312333131 
// 3: 13x (43%)
// 2: 4x  (14%)
// 1: 13x (43%)
// ReLU, l=0.3, w=[0.5,1]
// 1211111111 12111 ...
// 3: 0x (0%)
// 2: 2x (14%)
// 1: 13x (86%)
// ReLU, l=0.3, w=[0,0.5]
// 2232323223 2222222222 2222322222
// 3: 5x (16%)
// 2: 25x (84%)
// 1: 0x (0%)
// ReLU, l=0.3, w=[-0.5,0.5]
// 1331131133 1213131123 2131113321
// 3: 11x (37%)
// 2: 4x (13%)
// 1: 15x (50%)
// ReLU, l=0.3, w=[-1,0]
// 3111131131 3111111111 1213111113
// 3: 6x (20%)
// 2: 1x (3%)
// 1: 24x (77%)
// ReLU, l=0.3, w=[-1,1]
// 3321311311 1111111111 1111211111
// 3: 4x (13%)
// 2: 1x (3%)
// 1: 25x (84%)
function detect(network, data) {
    data.forEach(({ input, label }) => {
        const output = network.detect(input);
        const guess = Math.round(output[0].activation * 100) / 100;
        const err = Math.round(Math.abs(label - guess) * 100) / 100;
        console.log('guess: ' + guess.toFixed(2) + '      label: ' + label[0].toFixed(2) + '      error: ' + err.toFixed(2));
    });
}
function train(network, data, dataMultiplier, learningRate, weightFile) {
    const trainingData = shuffle(Array(dataMultiplier).fill(data).flat());
    trainingData.forEach(({ input, label }) => {
        network.train(input, label, learningRate);
    });
    console.log('Weights: ');
    console.log(network.weights);
    console.log('Errors: ');
    console.log(network.errors.join('\n'));
    network.writeWeightsToFile(weightFile);
}
function shuffle(array) {
    let currentIndex = array.length, randomIndex;
    while (currentIndex != 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
        [array[currentIndex], array[randomIndex]] = [
            array[randomIndex], array[currentIndex]
        ];
    }
    return array;
}
/*
   
   oo
   xx
   
   xx
   oo
   
   xo
   xo
   
   ox
   ox
   
   xo
   ox
   
   ox
   xo
   
   */ 
//# sourceMappingURL=main.js.map