"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const cnn_1 = require("./cnn");
// main()
mainRepeatedly();
function mainRepeatedly() {
    for (let i = 1; i < 31; i++) {
        console.log('\n' + i);
        main();
    }
}
function main() {
    const activation = 'sigmoid';
    const lower = 0;
    const higher = 1;
    const biasLower = -0.5;
    const biasHigher = 0.5;
    const learningRate = 0.3;
    const momentum = 0.8;
    const options = {
        activation: activation,
        weightOptions: {
            lower, higher, biasLower, biasHigher
        }
    };
    const MNISTCNN = new cnn_1.CNN(2, [2], 1, options);
    // console.log('initial weights: ')
    // console.log(JSON.stringify(MNISTCNN.beautifyWeights(MNISTCNN.weights)))
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
    train(MNISTCNN, dataXOR, 100000, learningRate, momentum);
    detect(MNISTCNN, dataXOR);
}
function detect(network, data) {
    data.forEach(({ input, label }) => {
        const output = network.detect(input);
        const guess = Math.round(output[0].activation * 100) / 100;
        const err = Math.round(Math.abs(label - guess) * 100) / 100;
        console.log(err);
        // console.log('guess: ' + guess.toFixed(2) + '      label: ' + label[0].toFixed(2) + '      error: ' + err.toFixed(2))
    });
}
function train(network, data, dataMultiplier, learningRate, monentum, weightFile) {
    const trainingData = shuffle(Array(dataMultiplier).fill(data).flat());
    trainingData.forEach(({ input, label }) => {
        network.train(input, label, learningRate, monentum);
    });
    // console.log('final weights: ')
    // console.log(JSON.stringify(network.beautifyWeights(network.weights)))
    // console.log('Errors: ')
    // console.log(network.errors.map((e: number) => e.toString().replace('.', ',')).join('\n'))
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