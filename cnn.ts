export class CNN {

    // notation: w_lij is weight i from neuron j in layer l
    weights: Weight[][][] = []
    // inputs: Layer
    outputs: Layer
    hiddenLayers: Layer[]
    activation: Activation

    constructor(inputs: number, hiddenLayers: number[], outputs: number, activation: ActivationType) {
        console.log('inputs: ' + inputs + ', hiddenLayers: ' + JSON.stringify(hiddenLayers) + ', outputs: ' + outputs)

        const initialNeuron: Neuron = { value: undefined }
        // this.inputs = Array(inputs).fill(initialNeuron)
        this.hiddenLayers = hiddenLayers.map((l): Layer => Array(l).fill(initialNeuron))
        this.outputs = Array(outputs).fill(initialNeuron)
        this.activation = this.activationFunctions[activation]

        const allLayers = [inputs, ...hiddenLayers, outputs]
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

    calculateNeuronValue(weights: Weight[], previousLayer: Layer) {
        if (weights.length !== previousLayer.length) {
            throw new Error('conflict at calculating next neuron value')
        }
        const numberOfWeights = weights.length
        let sumToReturn = 0
        for (let w = 0; w < numberOfWeights; w++) {
            sumToReturn += weights[w] * previousLayer[w].value
        }
        return sumToReturn
    }

    calculateOutput(inputs: Layer) {
        let previousLayer = inputs
        this.hiddenLayers.forEach((layer: Layer, l: number) => {
            console.log('layer ' + (l + 1) + ': ' + JSON.stringify(previousLayer))
            const newLayer = layer.map((neuron: Neuron, n: number) => {
                const relevantWeights = this.weights[l][n]
                const newValue = this.calculateNeuronValue(relevantWeights, previousLayer)
                const activatedValue = this.activation(newValue)
                return { ...neuron, value: activatedValue }
            })
            previousLayer = newLayer
        })

        const output = this.outputs.map((neuron: Neuron, n: number) => {
            const finalWeights = this.weights[this.weights.length - 1][n]
            return { value: this.activation(this.calculateNeuronValue(finalWeights, previousLayer)) }
        })
        console.log(output)
        return this.softmax(output)
    }

    softmax(layer: Layer) {
        const expsum = layer.reduce((i: number, { value }: Neuron) => i + Math.exp(value), 0)

        return layer.map(({ value }: Neuron) => {
            return Math.exp(value) / expsum
        })
    }

    train(inputs: Layer, label: Label) {
        const output = this.calculateOutput(inputs)
        const error = label - output
    }

    detect(inputs: Layer) {
        return this.calculateOutput(inputs)
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

}
/*
3 inputs to 5 neurons
[[w11 w12 w13 w13 w14 w15], [w21 w22 w23 w24 w25], [w31 w32 w33 w34 w35]]
5 neurons to 4 neurons
[[w11 w12 w13 w13 w14], [w21 w22 w23 w24], [w31 w32 w33 w34], [w41 w42 w43 w44], [w51 w52 w53 w54]]
4 neurons to 2 outputs
[[w11 w12], [w21 w22], [w31 w32], [w41 w42]]


inputs = 3
layers = [5, 4]
output = 2

[3, 5, 5, 4, 4, 2]



*/