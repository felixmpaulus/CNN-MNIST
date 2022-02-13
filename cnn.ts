export class CNN {

    // notation: w_lij is weight i from neuron j in layer l
    weights: Weight[][][] = []
    learningRate;
    output: Layer
    hiddenLayers: Layer[]
    activation: Activation

    constructor(input: number, hiddenLayers: number[], output: number, activation: ActivationType, learningRate: number) {
        console.log('input: ' + input + ', hiddenLayers: ' + JSON.stringify(hiddenLayers) + ', output: ' + output)

        const initialNeuron: Neuron = { value: undefined }
        // this.input = Array(input).fill(initialNeuron)
        this.hiddenLayers = hiddenLayers.map((l): Layer => Array(l).fill(initialNeuron))
        this.output = Array(output).fill(initialNeuron)
        this.activation = this.activationFunctions[activation]
        this.learningRate = learningRate

        const allLayers = [input, ...hiddenLayers, output]
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

    dotProduct(weights: Weight[], previousLayer: Layer) {
        if (weights.length !== previousLayer.length) {
            throw new Error('conflict at calculating next neuron value')
        }
        const numberOfWeights = weights.length
        let dotProductSum = 0
        for (let w = 0; w < numberOfWeights; w++) {
            dotProductSum += weights[w] * previousLayer[w].value
        }
        return dotProductSum
    }

    calculateOutput(input: Layer): Layer {
        let previousLayer = input
        this.hiddenLayers = this.hiddenLayers.map((layer: Layer, l: number): Layer => {
            console.log('layer ' + (l + 1) + ': ' + JSON.stringify(previousLayer))
            const newLayer = layer.map((neuron: Neuron, n: number) => {
                const relevantWeights = this.weights[l][n]
                const newValue = this.dotProduct(relevantWeights, previousLayer)
                const activatedValue = this.activation(newValue)
                return { ...neuron, value: activatedValue }
            })
            previousLayer = newLayer
            return newLayer
        })

        const output = this.output.map((neuron: Neuron, n: number) => {
            const finalWeights = this.weights[this.weights.length - 1][n]
            return { value: this.activation(this.dotProduct(finalWeights, previousLayer)) }
        })
        console.log(output)
        return this.softmax(output)
    }

    softmax(layer: Layer) {
        const expsum = layer.reduce((i: number, { value }: Neuron) => i + Math.exp(value), 0)

        return layer.map(({ value }: Neuron) => {
            return { value: (Math.exp(value) / expsum) }
        })
    }

    // the partial derivative of a weight is the product of:
    // error of node j in layer l * output of node i in layer l-1
    // with the error
    //  - beeing dependent of the values in the 'next' layer
    //  - using the values after passing through the activation function
    //  ...https://brilliant.org/wiki/backpropagation/


    train(input: Layer, label: Label) {
        const output = this.calculateOutput(input)
        const error = this.calculateError(output, label)
    }

    calculateError(output: Layer, label: Label) {
        if (output.length !== label.length) {
            throw new Error('conflict at calculating error')
        }
        const outputValues = output.map(o => o.value)
        const numberOfOutputs = output.length
        let squaredErrorSum = 0
        for (let o = 0; o < numberOfOutputs; o++) {
            squaredErrorSum += Math.pow(outputValues[o] - label[o], 2)
        }
        return (0.5 * squaredErrorSum) / numberOfOutputs
    }

    detect(input: Layer) {
        return this.calculateOutput(input)
    }

    getWeights() {
        console.log(this.weights)
    }

    activationFunctions = {
        'ReLU': this.activateReLU
    }

    activateReLU(value: number) {
        return value > 0 ? value : 0
    }

    deriviateReLU(value: number) {
        return value > 0 ? 1 : 0
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