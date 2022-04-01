import { activationFunctions } from './activation';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { moveMessagePortToContext } from 'worker_threads';

export class CNN {
    layers: number[]
    weights: Weights = [] // notation: w_lij is weight i from neuron j in layer l  
    previousDeltaWeights: Weights = []
    activation: Activation
    weightFile: string
    // debug
    errors: number[] = []

    constructor(inputSize: number, hiddenLayerSizes: number[], outputSize: number, options: NNOptions) {
        const { activation, weightOptions } = options
        this.activation = activationFunctions[activation]
        this.layers = [inputSize, ...hiddenLayerSizes, outputSize]
        this.weights = this.getInitialWeights(weightOptions)
        this.previousDeltaWeights = this.getInitialWeights({ fixedWeight: 0 })
    }

    detect(input: InputLayer): Layer {
        const activatedInputs = input.map(i => ({ activation: this.activation.primitive(i) }))
        const propagatedNetwork = this.propagate(activatedInputs)
        const outputLayer = propagatedNetwork[propagatedNetwork.length - 1]
        return outputLayer
    }

    train(input: InputLayer, label: Label, learningRate: number, momentum: number) {
        const activatedInputs = input.map(i => ({ activation: this.activation.primitive(i) }))
        const propagatedNetwork = this.propagate(activatedInputs)
        this.backpropagate(propagatedNetwork, label, learningRate, momentum)

        // debug
        const error = this.calculateError(propagatedNetwork[propagatedNetwork.length - 1], label)
        this.errors.push(error)
        // this.logAllValues(propagatedNetwork, label)
    }

    propagate(input: Layer): Layer[] {
        let previousLayer: Layer = input

        const initialNeuron: Neuron = { dotProduct: undefined, activation: undefined }
        let hiddenLayers = this.layers.slice(1).map((l): Layer => Array(l).fill(initialNeuron))

        hiddenLayers = hiddenLayers.map((layer: Layer, l: number): Layer => {
            const newLayer = layer.map((neuron: Neuron, n: number) => {
                const dotProduct = this.dotProduct(this.weights[l][n], previousLayer)
                const activation = this.activation.primitive(dotProduct)
                return { ...neuron, activation, dotProduct }
            })
            previousLayer = newLayer
            return newLayer
        })

        return [input, ...hiddenLayers]
    }

    backpropagate(propagatedNetwork: Layer[], label: Label, learningRate: number, momentum: number) {
        let previousSensitivities: number[] = []
        const oldWeights: Weights = this.deepClone(this.weights)
        for (let l = propagatedNetwork.length - 1; l >= 1; l--) {
            const isOutputLayer = l == propagatedNetwork.length - 1
            const layer = propagatedNetwork[l]
            const previousLayer = propagatedNetwork[l - 1]

            let sensitivities: number[] = []
            for (let n = 0; n < layer.length; n++) {
                const neuron = layer[n]
                const derivedActivation = this.activation.derivative(neuron.dotProduct)

                let derivedError
                if (isOutputLayer) {
                    derivedError = this.derivedError(neuron.activation, label[n])
                } else {
                    const weightsFromNeuron = oldWeights[l].reduce((p, m) => { return [...p, m.weights[n]] }, [])
                    derivedError = previousSensitivities.reduce((p, s, i) => { return p + s * weightsFromNeuron[i] }, 0)
                }

                let sensitivity = derivedError * derivedActivation
                sensitivities.push(sensitivity)

                const weightsToNeuron = oldWeights[l - 1][n].weights
                const biasToNeuron = oldWeights[l - 1][n].bias
                const newBiasDeltaWeight = - learningRate * sensitivity
                const previousBiasDeltaWeight = this.previousDeltaWeights[l - 1][n].bias
                this.weights[l - 1][n].bias = biasToNeuron + newBiasDeltaWeight + momentum * previousBiasDeltaWeight

                for (let w = 0; w < weightsToNeuron.length; w++) {
                    const weight = weightsToNeuron[w]
                    const { activation: leftConnectedActivation } = previousLayer[w]
                    const newDeltaWeight = - learningRate * sensitivity * leftConnectedActivation
                    const previousDeltaWeight = this.previousDeltaWeights[l - 1][n].weights[w]
                    const newWeight = weight + newDeltaWeight + momentum * previousDeltaWeight
                    this.previousDeltaWeights[l - 1][n].weights[w] = newDeltaWeight
                    this.weights[l - 1][n].weights[w] = newWeight
                }
            }
            previousSensitivities = sensitivities
            sensitivities = []
        }
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

    dotProduct({ weights, bias }: { weights: Weight[], bias: Bias }, previousLayer: Layer) {

        if (weights.length !== previousLayer.length) {
            throw new Error('conflict at calculating next neuron value')
        }
        const numberOfWeights = weights.length
        let dotProductSum = bias
        for (let w = 0; w < numberOfWeights; w++) {
            dotProductSum += weights[w] * previousLayer[w].activation
        }
        return dotProductSum
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

    weightFileExists(weightFile: string): Boolean {
        const path = 'weights/' + weightFile + '.json'
        return existsSync(path)
    }

    readWeightFile(weightFile: string) {
        const path = 'weights/' + weightFile + '.json'
        const weightFileContent = readFileSync(path, { encoding: 'utf8' })
        return JSON.parse(weightFileContent)
    }

    hasIdenticalDimensions(layers: number[]) {
        return layers.length === this.layers.length && layers.every((l, i) => {
            return l === this.layers[i]
        })
    }

    writeWeightsToFile(weightFile: string) {
        if (weightFile || this.weightFile) {
            const path = 'weights/' + (weightFile || this.weightFile) + '.json'
            const content = { layers: this.layers, weights: this.weights }
            writeFileSync(path, JSON.stringify(content))
        }
    }

    getInitialWeights(weightOptions?: WeightOptions): Weights {
        const { weightFile, fixedWeights, fixedWeight, lower, higher, biasLower, biasHigher } = weightOptions || {}
        this.weightFile = weightFile
        let initialWeights
        if (weightFile && this.weightFileExists(weightFile)) {
            const { layers, weights } = this.readWeightFile(weightFile)
            if (this.hasIdenticalDimensions(layers)) {
                initialWeights = weights
            } else {
                initialWeights = this.getRandomWeights({ lower, higher, biasLower, biasHigher })
            }
        } else if (fixedWeights) {
            initialWeights = fixedWeights
        } else if (typeof fixedWeight !== 'undefined') {
            initialWeights = this.getRandomWeights({ fixedWeight })
        } else {
            initialWeights = this.getRandomWeights({ lower, higher, biasLower, biasHigher })
        }
        return initialWeights
    }

    getRandomWeights(weightOptions: WeightOptions): Weights {
        const { fixedWeight, lower, higher, biasLower, biasHigher } = weightOptions
        if (lower >= higher || (typeof lower !== 'undefined' && (typeof higher === "undefined"))) {
            throw new Error('cant initialize normal weights')
        }
        if (biasLower >= biasHigher || (typeof biasLower !== 'undefined' && (typeof biasHigher === "undefined"))) {
            throw new Error('cant initialize bias weights')
        }
        const layerDimensions = this.layers.reduce((p, l, i) => (this.layers[i + 1] ? [...p, [l, this.layers[i + 1]]] : p), [])

        const initialWeights: Weights = []

        layerDimensions.forEach(([neurons, weights]) => {
            const weightsForLayers: { weights: Weight[], bias: Bias }[] = []
            for (let w = 0; w < weights; w++) {

                const bias: Bias = typeof fixedWeight !== 'undefined' ? fixedWeight : Math.random() * ((biasHigher || 1) - (biasLower || 0)) + (biasLower || 0)
                const weightsToNeuron: { weights: Weight[], bias: Bias } = { weights: [], bias }
                for (let n = 0; n < neurons; n++) {
                    const randomWeight = typeof fixedWeight !== 'undefined' ? fixedWeight : Math.random() * ((higher || 1) - (lower || 0)) + (lower || 0)
                    weightsToNeuron.weights.push(randomWeight)
                }
                weightsForLayers.push(weightsToNeuron)
            }
            initialWeights.push(weightsForLayers)
        })
        return initialWeights
    }

    beautifyWeights(weights: Weights) {
        return weights.map(l => {
            return l.map(n => {
                return {
                    weights: n.weights.map(w => {
                        return Math.round(w * 100) / 100
                    }),
                    bias: Math.round(n.bias * 100) / 100
                }
            })
        })
    }
}