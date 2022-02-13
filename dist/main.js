"use strict";
// import * as fs from 'fs';
Object.defineProperty(exports, "__esModule", { value: true });
const cnn_1 = require("./cnn");
main();
function main() {
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
    const testData = [
        { input: [0, 0, 1, 1], label: 1 },
        { input: [1, 1, 0, 0], label: 1 },
        { input: [1, 0, 1, 0], label: 1 },
        { input: [0, 1, 0, 1], label: 1 },
        { input: [1, 0, 0, 1], label: 0 },
        { input: [0, 1, 1, 0], label: 0 }
    ];
    // const MNISTasJSON = convertMNISTtoJSON()
    const MNISTCNN = new cnn_1.CNN(4, [4], 1, 'ReLU', 1);
    testData.forEach((t) => {
        const input = t.input.map((i) => ({ value: i }));
        const output = MNISTCNN.calculateOutput(input);
        console.log(output);
    });
    // const input: Layer = [{ value: 3 }, { value: 6 }]
    // console.log('output: ' + JSON.stringify(output))
    // MNISTCNN.getWeights()
}
// function convertMNISTtoJSON() {
//     var dataFileBuffer = fs.readFileSync(__dirname + '/MNIST/train-images-idx3-ubyte');
//     var labelFileBuffer = fs.readFileSync(__dirname + '/MNIST/train-labels-idx1-ubyte');
//     var pixelValues = [];
//     for (var image = 0; image <= 59999; image++) {
//         var pixels = [];
//         for (var x = 0; x <= 27; x++) {
//             for (var y = 0; y <= 27; y++) {
//                 pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 15]);
//             }
//         }
//         var imageData = {};
//         imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixels;
//         pixelValues.push(imageData);
//     }
//     // There are 28x28=784 pixel values, all varying from 0 to 255.
//     // [
//     //     { 5: [28, 0, 0, 0, 0, 0, 0, 0, 0, 0...] },
//     //     { 0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0...] },
//     //     ...
//     // ]
//     return pixelValues
// }
//# sourceMappingURL=main.js.map