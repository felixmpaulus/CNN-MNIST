"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CNN = void 0;
const activation_1 = require("./activation");
class CNN {
    constructor(inputSize, hiddenLayerSizes, outputSize, activation) {
        this.weights = [];
        console.log('input: ' + inputSize + ', hiddenLayers: ' + JSON.stringify(hiddenLayerSizes) + ', output: ' + outputSize);
        this.activation = activation_1.activationFunctions[activation];
        this.layers = [inputSize, ...hiddenLayerSizes, outputSize];
        const layerDimensions = this.layers.reduce((p, l, i) => (this.layers[i + 1] ? [...p, [l, this.layers[i + 1]]] : p), []);
        console.log('Layer Sizes: ' + JSON.stringify(layerDimensions));
        layerDimensions.forEach(([neurons, weights]) => {
            const weightsForNeuron = [];
            for (let w = 0; w < weights; w++) {
                const weightsForLayers = [];
                for (let n = 0; n < neurons; n++) {
                    weightsForLayers.push(Math.random());
                }
                weightsForNeuron.push(weightsForLayers);
            }
            this.weights.push(weightsForNeuron);
        });
        console.log('Weights: ');
        console.log(this.weights);
    }
    detect(input) {
        return this.calculateOutput(input);
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
    calculateOutput(input) {
        let previousLayer = input.map(i => ({ activation: i }));
        const initialNeuron = { dotProduct: undefined, activation: undefined };
        let hiddenLayers = this.layers.slice(1).map((l) => Array(l).fill(initialNeuron));
        hiddenLayers = hiddenLayers.map((layer, l) => {
            const newLayer = layer.map((neuron, n) => {
                const relevantWeights = this.weights[l][n];
                const dotProduct = this.dotProduct(relevantWeights, previousLayer);
                const activation = this.activation.primitive(dotProduct);
                return Object.assign(Object.assign({}, neuron), { activation, dotProduct });
            });
            previousLayer = newLayer;
            return newLayer;
        });
        this.logAllValues(input, hiddenLayers);
        return hiddenLayers[hiddenLayers.length - 1];
    }
    train(input, label, learningRate) {
        const output = this.calculateOutput(input);
        const error = this.calculateError(output, label);
        console.log('output: ' + output.map(o => o.activation) + ', label: ' + label + ', error: ' + error);
    }
    calculateError(output, label) {
        if (output.length !== label.length) {
            throw new Error('conflict at calculating error');
        }
        const outputActivations = output.map(o => o.activation);
        const numberOfOutputs = output.length;
        let squaredErrorSum = 0;
        for (let o = 0; o < numberOfOutputs; o++) {
            squaredErrorSum += this.halfSquaredError(outputActivations[o], label[o]);
        }
        return squaredErrorSum / numberOfOutputs;
    }
    // del_E / del_w_i
    partialDerivativeErrorFinalLayer(real, desired, dotProduct, previousNode) {
        const { activation } = previousNode;
        return this.derivedError(real, desired, dotProduct) * activation;
    }
    // E
    halfSquaredError(real, desired) {
        return 0.5 * (Math.pow(real - desired, 2));
    }
    // δ
    derivedError(real, desired, dotProduct) {
        return (real - desired) * this.activation.derivative(dotProduct);
    }
    partialDerivativeErrorHiddenLayer(real, desired, dotProduct, previousNode) {
    }
    dotProduct(weights, previousLayer) {
        if (weights.length !== previousLayer.length) {
            throw new Error('conflict at calculating next neuron value');
        }
        const numberOfWeights = weights.length;
        let dotProductSum = 0;
        for (let w = 0; w < numberOfWeights; w++) {
            dotProductSum += weights[w] * previousLayer[w].activation;
        }
        return dotProductSum;
    }
    softmax(layer) {
        const expsum = layer.reduce((i, { activation }) => i + Math.exp(activation), 0);
        return layer.map((neuron) => {
            const { activation } = neuron;
            return Object.assign({ activation: (Math.exp(activation) / expsum) }, neuron);
        });
    }
    logAllValues(input, hiddenLayers) {
        let log = '';
        const inputAsStandardLayer = input.map(i => ({ activation: i, dotProduct: undefined, error: undefined }));
        const allLayers = [inputAsStandardLayer, ...hiddenLayers];
        let i = 0;
        while (true) {
            let continueAdding = false;
            allLayers.forEach((l) => {
                const neuron = l[i];
                log = log + (neuron ? (Math.round(neuron.activation * 100) / 100) : ' ') + '    ';
                continueAdding = (continueAdding || !!neuron);
            });
            if (!continueAdding)
                break;
            log += '\n';
            i += 1;
        }
        console.log(log);
    }
}
exports.CNN = CNN;
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
//# sourceMappingURL=cnn.js.map