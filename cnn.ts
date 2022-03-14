import { activationFunctions } from './activation';

export class CNN {

    // notation: w_lij is weight i from neuron j in layer l
    layers: number[]
    weights: Weight[][][] = []
    activation: Activation

    constructor(inputSize: number, hiddenLayerSizes: number[], outputSize: number, activation: ActivationType) {
        console.log('input: ' + inputSize + ', hiddenLayers: ' + JSON.stringify(hiddenLayerSizes) + ', output: ' + outputSize)

        this.activation = activationFunctions[activation]

        this.layers = [inputSize, ...hiddenLayerSizes, outputSize]
        const layerDimensions = this.layers.reduce((p, l, i) => (this.layers[i + 1] ? [...p, [l, this.layers[i + 1]]] : p), [])
        console.log('Layer Sizes: ' + JSON.stringify(layerDimensions))

        layerDimensions.forEach(([neurons, weights]) => {
            const weightsForNeuron: Weight[][] = []
            for (let w = 0; w < weights; w++) {
                const weightsForLayers: Weight[] = []
                for (let n = 0; n < neurons; n++) {
                    weightsForLayers.push(Math.random())
                }
                weightsForNeuron.push(weightsForLayers)
            }
            this.weights.push(weightsForNeuron)
        })
        console.log('Weights: ')
        console.log(this.weights)
    }

    detect(input: InputLayer): Layer {
        const propagatedNetwork = this.propagate(input)
        const outputLayer = propagatedNetwork[propagatedNetwork.length - 1]
        return outputLayer
    }

    propagate(input: InputLayer): Layer[] {
        const inputAsStandardLayer = input.map(i => ({ activation: i }))
        let previousLayer: Layer = inputAsStandardLayer

        const initialNeuron: Neuron = { dotProduct: undefined, activation: undefined }
        let hiddenLayers = this.layers.slice(1).map((l): Layer => Array(l).fill(initialNeuron))

        hiddenLayers = hiddenLayers.map((layer: Layer, l: number): Layer => {
            const newLayer = layer.map((neuron: Neuron, n: number) => {
                const relevantWeights = this.weights[l][n]
                const dotProduct = this.dotProduct(relevantWeights, previousLayer)
                const activation = this.activation.primitive(dotProduct)
                return { ...neuron, activation, dotProduct }
            })
            previousLayer = newLayer
            return newLayer
        })

        this.logAllValues(input, hiddenLayers)
        return [inputAsStandardLayer, ...hiddenLayers]
    }

    train(input: InputLayer, label: Label, learningRate: number) {
        const propagatedNetwork = this.propagate(input)
        for (let l = propagatedNetwork.length - 1; l > 1; l--) {
            // iterate backwards through all weights

            // output layer
            // calculate the new weight by
            // w = w - l dE/dw
            // dE/dw = derived Error * derived activation(value) * output previous neuron(activation)

            // hidden layer 1
            // calculate the new weight by
            // w = w - l * dE/dw
            // dE/dw = magic * derived activation(value) * output previous neuron(activation)
            // magic = complete error function derived by the activation of the neuron the weight goes to
            //         wich is calculated by the sum of all errors derived by the activation

            // the del for a neuron in the previous layer del_j is calculated by
            // sum over all neurons in the current layer:
            //      the new weight going from the neuron the previous layer to the neuron
            //      
            //      the derived activation function of the neuron
        }





        const outputLayer = propagatedNetwork[propagatedNetwork.length - 1]
        const error = this.calculateError(outputLayer, label)

        console.log('output: ' + outputLayer.map(o => o.activation) + ', label: ' + label + ', error: ' + error)

    }

    calculateError(output: Layer, label: Label) {
        if (output.length !== label.length) {
            throw new Error('conflict at calculating error')
        }
        const outputActivations = output.map(o => o.activation)
        const numberOfOutputs = output.length
        let squaredErrorSum = 0
        for (let o = 0; o < numberOfOutputs; o++) {
            squaredErrorSum += this.halfSquaredError(outputActivations[o], label[o])
        }
        return squaredErrorSum / numberOfOutputs
    }

    halfSquaredError(real: number, desired: number): number {
        return 0.5 * (Math.pow(real - desired, 2))
    }

    derivedError(real: number, desired: number): number {
        return (real - desired)
    }

    dotProduct(weights: Weight[], previousLayer: Layer) {
        if (weights.length !== previousLayer.length) {
            throw new Error('conflict at calculating next neuron value')
        }
        const numberOfWeights = weights.length
        let dotProductSum = 0
        for (let w = 0; w < numberOfWeights; w++) {
            dotProductSum += weights[w] * previousLayer[w].activation
        }
        return dotProductSum
    }

    softmax(layer: Layer): Layer {
        const expsum = layer.reduce((i: number, { activation }: Neuron) => i + Math.exp(activation), 0)
        return layer.map((neuron: Neuron) => {
            const { activation } = neuron
            return { activation: (Math.exp(activation) / expsum), ...neuron }
        })
    }

    logAllValues(input: InputLayer, hiddenLayers: Layer[]) {
        let log = ''
        const inputAsStandardLayer: Layer = input.map(i => ({ activation: i, dotProduct: undefined, error: undefined }))
        const allLayers: Layer[] = [inputAsStandardLayer, ...hiddenLayers]
        let i = 0
        while (true) {
            let continueAdding: Boolean = false
            allLayers.forEach((l: Layer) => {
                const neuron = l[i]
                log = log + (neuron ? (Math.round(neuron.activation * 100) / 100) : ' ') + '    '
                continueAdding = (continueAdding || !!neuron)
            })
            if (!continueAdding) break
            log += '\n'
            i += 1
        }
        console.log(log)
    }

}


/*
3 input to 5 neurons
[[w11 w12 w13 w13 w14 w15], [w21 w22 w23 w24 w25], [w31 w32 w33 w34 w35]]
5 neurons to 4 neurons
[[w11 w12 w13 w13 w14], [w21 w22 w23 w24], [w31 w32 w33 w34], [w41 w42 w43 w44], [w51 w52 w53 w54]]
4 neurons to 2 output
[[w11 w12], [w21 w22], [w31 w32], [w41 w42]]


input = 3
layers = [5, 4]
output = 2

[3, 5, 5, 4, 4, 2]



*/