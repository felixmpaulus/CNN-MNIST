"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const cnn_1 = require("./cnn");
main();
function main() {
    const weightFile = 'squares2';
    const MNISTCNN = new cnn_1.CNN(4, [2], 1, 'ReLU', weightFile);
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
    // train(MNISTCNN, data4Squares, weightFile)
    detect(MNISTCNN, data4Squares);
}
function detect(network, data) {
    data.forEach(({ input, label }) => {
        const output = network.detect(input);
        const guess = Math.round(output[0].activation * 100) / 100;
        const err = Math.round(Math.abs(label - guess) * 100) / 100;
        console.log('guess: ' + guess.toFixed(2) + '      label: ' + label[0].toFixed(2) + '      error: ' + err.toFixed(2));
    });
}
function train(network, data, weightFile) {
    const trainingData = shuffle(Array(100).fill(data).flat());
    trainingData.forEach(({ input, label }) => {
        network.train(input, label, 0.3);
    });
    console.log('Weights: ');
    console.log(network.weights);
    console.log('Errors: ');
    console.log(network.errors.join('\n'));
    console.log('Outputs: ');
    console.log(network.outputs.join('\n'));
    network.writeWeights(weightFile);
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