import { CNN } from './cnn'
import generateData from './dataGenerator'
main()
// mainRepeatedly()

// function mainRepeatedly() {
//     for (let i = 1; i < 31; i++) {
//         console.log('\n' + i)
//         main()
//     }
// }

function main() {
    const activation = 'sigmoid'
    const lower = 0
    const higher = 1
    const biasLower = -0.5
    const biasHigher = 0.5
    const learningRate = 0.03
    const momentum = 0.8
    const options: NNOptions = {
        activation: activation,
        weightOptions: {
            // lower, higher, biasLower, biasHigher
        }
    }

    const NN = new CNN(2, [4], 1, options)
    console.log(NN.weights[0])
    console.log(NN.weights[1])
    const data4Areas = generateData({ numberOfDatapoints: 1000, xMax: 10, xMin: 0, yMax: 10, yMin: 0, divider: 5 })
    const training = data4Areas.slice(0, 799)
    const validation = data4Areas.slice(800, 999)

    train(NN, training, 10000, learningRate, momentum)
    detect(NN, validation)
}

function train(network: any, data: any[], epochs: number, learningRate: number, monentum: number, weightFile?: string) {

    const trainingData = shuffle(Array(epochs).fill(data).flat())

    trainingData.forEach(({ input, label }) => {
        network.train(input, label, learningRate, monentum)
    })

    // console.log('final weights: ')
    // console.log(JSON.stringify(network.beautifyWeights(network.weights)))
    console.log('Errors: ')
    console.log(network.errors.map((e: number) => e.toString().replace('.', ',')).join('\n'))
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

const dataXOR = [
    { input: [1, 1], label: [0] },
    { input: [0, 1], label: [1] },
    { input: [1, 0], label: [1] },
    { input: [0, 0], label: [0] },]

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