import { activationFunctions } from './activation';

export class CNN {

    // notation: w_lij is weight i from neuron j in layer l
    weights: Weight[][][] = []
    learningRate;
    output: Layer
    hiddenLayers: Layer[]
    activation: Activation

    constructor(inputSize: number, hiddenLayerSizes: number[], outputSize: number, activation: ActivationType, learningRate: number) {
        console.log('input: ' + inputSize + ', hiddenLayers: ' + JSON.stringify(hiddenLayerSizes) + ', output: ' + outputSize)

        const initialNeuron: Neuron = { dotProduct: undefined, activation: undefined, error: undefined }
        this.hiddenLayers = hiddenLayerSizes.map((l): Layer => Array(l).fill(initialNeuron))
        this.output = Array(outputSize).fill(initialNeuron)
        this.activation = activationFunctions[activation]
        this.learningRate = learningRate

        const allLayers = [inputSize, ...hiddenLayerSizes, outputSize]
        const layerDimensions = allLayers.reduce((p, l, i) => (allLayers[i + 1] ? [...p, [l, allLayers[i + 1]]] : p), [])
        console.log('Layer Dimensions: ' + JSON.stringify(layerDimensions))

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

    detect(input: InputLayer) {
        return this.calculateOutput(input)
    }

    // we go from the back to the front
    // final layer:
    // we deriviate the full error function by one weight after another (all weights in total)
    // d_Error / d_dotProduct_j * d_dotProduct_j / d_w_ij
    // the partial derivative of a weight is the product of:
    // error of node j in layer l * output of node i in layer l-1
    // with the error
    //  - beeing dependent of the values in the 'next' layer
    //  - using the values after passing through the activation function


    train(input: InputLayer, label: Label) {
        const output = this.calculateOutput(input)
        const error = this.calculateError(output, label)
    }

    calculateOutput(input: InputLayer): Layer {
        let previousLayer: Layer = input.map(i => ({ activation: i, dotProduct: undefined, error: undefined }))
        this.hiddenLayers = this.hiddenLayers.map((layer: Layer, l: number): Layer => {
            const newLayer = layer.map((neuron: Neuron, n: number) => {
                const relevantWeights = this.weights[l][n]
                const dotProduct = this.dotProduct(relevantWeights, previousLayer)
                const activation = this.activation.primitive(dotProduct)
                return { ...neuron, activation, dotProduct }
            })
            previousLayer = newLayer
            return newLayer
        })

        this.output = this.output.map((neuron: Neuron, n: number) => {
            const finalWeights = this.weights[this.weights.length - 1][n]
            const dotProduct = this.dotProduct(finalWeights, previousLayer)
            const activation = this.activation.primitive(dotProduct)
            return { ...neuron, activation, dotProduct }
        })

        return this.output
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

    // del_E / del_w_i
    partialDerivativeErrorFinalLayer(real: number, desired: number, dotProduct: number, previousNode: Neuron) {
        const { activation } = previousNode
        return this.derivedError(real, desired, dotProduct) * activation
    }

    // E
    halfSquaredError(real: number, desired: number): number {
        return 0.5 * (Math.pow(real - desired, 2))
    }

    // δ
    derivedError(real: number, desired: number, dotProduct: number): number {
        return (real - desired) * this.activation.derivative(dotProduct)
    }

    partialDerivativeErrorHiddenLayer(real: number, desired: number, dotProduct: number, previousNode: Neuron) {

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

    logAllVAlues(input: InputLayer) {
        let log = ''
        const inputAsStandardLayer: Layer = input.map(i => ({ activation: i, dotProduct: undefined, error: undefined }))
        const allLayers: Layer[] = [inputAsStandardLayer, ...this.hiddenLayers, this.output]
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