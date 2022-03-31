import { deepStrictEqual } from "assert"
import { CNN } from "./cnn"

main()
// mainRepeatedly()

// function mainRepeatedly() {
//     for (let i = 1; i < 31; i++) {
//         console.log('\n' + i)
//         main()
//     }
// }

function main() {
    const activation = 'leakyReLU'
    const lowerLimit = 0
    const higherLimit = 1
    const learningRate = 0.01
    const options: NNOptions = {
        activation: activation,
        weightOptions: {
            lowerLimit, higherLimit
        }
    }
    const MNISTCNN = new CNN(2, [2], 1, options)
    // console.log('initial weights: ')
    // console.log(JSON.stringify(MNISTCNN.beautifyWeights(MNISTCNN.weights)))

    const data4Squares = [
        { input: [0, 0, 1, 1], label: [1] },
        { input: [1, 1, 0, 0], label: [1] },
        { input: [1, 0, 1, 0], label: [1] },
        { input: [0, 1, 0, 1], label: [1] },
        { input: [1, 0, 0, 1], label: [0] },
        { input: [0, 1, 1, 0], label: [0] }
    ]

    const dataXOR = [
        { input: [1, 1], label: [0] },
        { input: [0, 1], label: [1] },
        { input: [1, 0], label: [1] },
        { input: [0, 0], label: [0] },]


    train(MNISTCNN, dataXOR, 15000, learningRate)
    detect(MNISTCNN, dataXOR)
}


function detect(network: any, data: any[]) {
    data.forEach(({ input, label }) => {
        const output = network.detect(input)
        const guess = Math.round(output[0].activation * 100) / 100
        const err = Math.round(Math.abs(label - guess) * 100) / 100
        console.log(err)
        // console.log('guess: ' + guess.toFixed(2) + '      label: ' + label[0].toFixed(2) + '      error: ' + err.toFixed(2))
    })
}

function train(network: any, data: any[], dataMultiplier: number, learningRate: number, weightFile?: string) {

    const trainingData = shuffle(Array(dataMultiplier).fill(data).flat())

    trainingData.forEach(({ input, label }) => {
        network.train(input, label, learningRate)
    })

    // console.log('final weights: ')
    // console.log(JSON.stringify(network.beautifyWeights(network.weights)))
    console.log('Errors: ')
    console.log(network.errors.map((e: number) => e.toString().replace('.', ',')).join('\n'))

    network.writeWeightsToFile(weightFile)
}

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