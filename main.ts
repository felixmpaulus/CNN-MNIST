// import * as fs from 'fs';

import { CNN } from "./cnn"

main()

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
        // { input: [0, 0, 1, 1], label: [1] },
        // { input: [1, 1, 0, 0], label: [1] },
        // { input: [1, 0, 1, 0], label: [1] },
        { input: [0, 1, 0, 1], label: [1] },
        { input: [1, 0, 0, 1], label: [0] },
        // { input: [0, 1, 1, 0], label: [0] }
    ]

    const MNISTCNN = new CNN(4, [4], 1, 'ReLU')

    const multipleTestData = [...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData, ...testData]
    shuffle(multipleTestData).forEach(({ input, label }) => {
        MNISTCNN.train(input, label, 0.1)
    })



    console.log('Weights: ')
    console.log(MNISTCNN.weights)
    console.log('Errors: ')
    console.log(MNISTCNN.errors.join(',\n'))
    console.log('Outputs: ')
    console.log(MNISTCNN.outputs.join(',\n'))

    const output1 = MNISTCNN.detect([1, 0, 0, 1])
    console.log('output1: ' + output1[0].activation)
    const output2 = MNISTCNN.detect([0, 1, 0, 1])
    console.log('output2: ' + output2[0].activation)













    function shuffle(array: any[]) {
        let currentIndex = array.length, randomIndex;
        while (currentIndex != 0) {
            randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex--;
            [array[currentIndex], array[randomIndex]] = [
                array[randomIndex], array[currentIndex]];
        }
        return array;
    }
}



