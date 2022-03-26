import { activationFunctions } from './activation';

export class CNN {


    layers: number[]
    weights: Weight[][][] = [] // notation: w_lij is weight i from neuron j in layer l 
    // weights = [
    //     [
    //         [0.33, 0.64, 0.95, 0.91],
    //         [0.33, 0.73, 0.58, 0.05],
    //         [0.4, 0.63, 0.64, 0.06],
    //         [0.83, 0.75, 0.39, 0.33]
    //     ],
    //     [[0.88, 0.27, 0.44, 0.61]]
    // ]
    activation: Activation
    // debug
    errors: number[] = []
    outputs: number[] = []

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
                    weightsForLayers.push(Math.round(Math.random() * 100) / 100)
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

    train(input: InputLayer, label: Label, learningRate: number) {
        const propagatedNetwork = this.propagate(input)
        this.backpropagate(propagatedNetwork, label, learningRate)
        const error = this.calculateError(propagatedNetwork[propagatedNetwork.length - 1], label)
        this.errors.push(error)
        const output = propagatedNetwork[propagatedNetwork.length - 1][0].activation
        this.outputs.push(output)
        this.logAllValues(propagatedNetwork, label)
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

        return [inputAsStandardLayer, ...hiddenLayers]
    }

    backpropagate(propagatedNetwork: Layer[], label: Label, learningRate: number) {
        let previousSensitivities: number[] = []
        const oldWeights: number[][][] = this.deepClone(this.weights)
        for (let l = propagatedNetwork.length - 1; l >= 1; l--) {
            console.log('propagating from layer: ' + l + ' to layer: ' + (l - 1))
            const isOutputLayer = l == propagatedNetwork.length - 1
            const layer = propagatedNetwork[l]
            const previousLayer = propagatedNetwork[l - 1]

            let sensitivities: number[] = []
            for (let n = 0; n < layer.length; n++) {
                console.log('looking at neuron: ' + n)
                const neuron = layer[n]
                const weightsToNeuron = oldWeights[l - 1][n]

                const derivedActivation = this.activation.derivative(neuron.dotProduct)

                let derivedError
                if (isOutputLayer) {
                    derivedError = this.derivedError(neuron.activation, label[n])
                } else {
                    const weightsFromNeuron = oldWeights[l].reduce((p, m) => { return [...p, m[n]] }, [])
                    derivedError = previousSensitivities.reduce((p, s, i) => { return p + s * weightsFromNeuron[i] }, 0)
                }

                let sensitivity = derivedError * derivedActivation
                sensitivities.push(sensitivity)

                for (let w = 0; w < weightsToNeuron.length; w++) {
                    const weight = weightsToNeuron[w]
                    const { activation: leftConnectedActivation } = previousLayer[w]
                    const newWeight = weight - learningRate * sensitivity * leftConnectedActivation
                    console.log('updating w_' + w + n + '. weight was: ' + weight + '. new weight is: ' + newWeight)
                    this.weights[l - 1][n][w] = newWeight
                }
            }
            console.log('sensitivities: ')
            console.log(sensitivities)
            previousSensitivities = sensitivities
            sensitivities = []
        }

        return
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

    logAllValues(propagatedNetwork: Layer[], label: Label) {
        let log = '\n\n- - - - - - - - - - -\nneuron activation values:\n'
        let i = 0
        while (true) {
            let continueAdding: Boolean = false
            propagatedNetwork.forEach((l: Layer) => {
                const neuron = l[i]
                log = log + (neuron ? (Math.round(neuron.activation * 100) / 100) : ' ') + '    '
                continueAdding = (continueAdding || !!neuron)
            })
            if (!continueAdding) break
            log += '\n'
            i += 1
        }
        console.log(log)

        const outputLayer = propagatedNetwork[propagatedNetwork.length - 1]
        const error = this.calculateError(outputLayer, label)

        console.log('output: ' + outputLayer.map(o => Math.round(o.activation * 1000) / 1000) + ', \nlabel: ' + label + ', \nerror: ' + Math.round(error * 1000) / 1000 + '\n- - - - - - - - - - -\n\n')
    }

    deepClone(object: Object) {
        return JSON.parse(JSON.stringify(object))
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