"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CNN = void 0;
const activation_1 = require("./activation");
const fs_1 = require("fs");
class CNN {
    constructor(inputSize, hiddenLayerSizes, outputSize, activation, weightFile) {
        this.weights = []; // notation: w_lij is weight i from neuron j in layer l 
        // debug
        this.errors = [];
        this.outputs = [];
        console.log('input: ' + inputSize + ', hiddenLayers: ' + JSON.stringify(hiddenLayerSizes) + ', output: ' + outputSize);
        this.activation = activation_1.activationFunctions[activation];
        this.weightFile = weightFile;
        this.layers = [inputSize, ...hiddenLayerSizes, outputSize];
        const fileExists = weightFile && this.weightFileExists(weightFile);
        if (fileExists) {
            const { layers, weights } = this.readWeightFile(weightFile);
            if (this.hasIdenticalDimensions(layers)) {
                this.weights = weights;
                console.log('loaded weightFile: ' + weightFile);
            }
            else {
                this.assignRandomWeights();
            }
        }
        else {
            this.assignRandomWeights();
        }
    }
    assignRandomWeights() {
        const layerDimensions = this.layers.reduce((p, l, i) => (this.layers[i + 1] ? [...p, [l, this.layers[i + 1]]] : p), []);
        console.log('Layer Sizes: ' + JSON.stringify(layerDimensions));
        layerDimensions.forEach(([neurons, weights]) => {
            const weightsForNeuron = [];
            for (let w = 0; w < weights; w++) {
                const weightsForLayers = [];
                for (let n = 0; n < neurons; n++) {
                    weightsForLayers.push(Math.round(Math.random() * 100) / 100);
                }
                weightsForNeuron.push(weightsForLayers);
            }
            this.weights.push(weightsForNeuron);
        });
        console.log('assigned random weights');
    }
    detect(input) {
        const propagatedNetwork = this.propagate(input);
        const outputLayer = propagatedNetwork[propagatedNetwork.length - 1];
        return outputLayer;
    }
    train(input, label, learningRate) {
        const propagatedNetwork = this.propagate(input);
        this.backpropagate(propagatedNetwork, label, learningRate);
        const error = this.calculateError(propagatedNetwork[propagatedNetwork.length - 1], label);
        this.errors.push(error);
        const output = propagatedNetwork[propagatedNetwork.length - 1][0].activation;
        this.outputs.push(output);
        this.logAllValues(propagatedNetwork, label);
    }
    propagate(input) {
        const inputAsStandardLayer = input.map(i => ({ activation: i }));
        let previousLayer = inputAsStandardLayer;
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
        return [inputAsStandardLayer, ...hiddenLayers];
    }
    backpropagate(propagatedNetwork, label, learningRate) {
        let previousSensitivities = [];
        const oldWeights = this.deepClone(this.weights);
        for (let l = propagatedNetwork.length - 1; l >= 1; l--) {
            // console.log('propagating from layer: ' + l + ' to layer: ' + (l - 1))
            const isOutputLayer = l == propagatedNetwork.length - 1;
            const layer = propagatedNetwork[l];
            const previousLayer = propagatedNetwork[l - 1];
            let sensitivities = [];
            for (let n = 0; n < layer.length; n++) {
                // console.log('looking at neuron: ' + n)
                const neuron = layer[n];
                const weightsToNeuron = oldWeights[l - 1][n];
                const derivedActivation = this.activation.derivative(neuron.dotProduct);
                let derivedError;
                if (isOutputLayer) {
                    derivedError = this.derivedError(neuron.activation, label[n]);
                }
                else {
                    const weightsFromNeuron = oldWeights[l].reduce((p, m) => { return [...p, m[n]]; }, []);
                    derivedError = previousSensitivities.reduce((p, s, i) => { return p + s * weightsFromNeuron[i]; }, 0);
                }
                let sensitivity = derivedError * derivedActivation;
                sensitivities.push(sensitivity);
                for (let w = 0; w < weightsToNeuron.length; w++) {
                    const weight = weightsToNeuron[w];
                    const { activation: leftConnectedActivation } = previousLayer[w];
                    const newWeight = weight - learningRate * sensitivity * leftConnectedActivation;
                    // console.log('updating w_' + w + n + '. weight was: ' + weight + '. new weight is: ' + newWeight)
                    this.weights[l - 1][n][w] = newWeight;
                }
            }
            // console.log('sensitivities: ')
            // console.log(sensitivities)
            previousSensitivities = sensitivities;
            sensitivities = [];
        }
        return;
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
    halfSquaredError(real, desired) {
        return 0.5 * (Math.pow(real - desired, 2));
    }
    derivedError(real, desired) {
        return (real - desired);
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
    logAllValues(propagatedNetwork, label) {
        let log = '\n\n- - - - - - - - - - -\nneuron activation values:\n';
        let i = 0;
        while (true) {
            let continueAdding = false;
            propagatedNetwork.forEach((l) => {
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
        const outputLayer = propagatedNetwork[propagatedNetwork.length - 1];
        const error = this.calculateError(outputLayer, label);
        console.log('output: ' + outputLayer.map(o => Math.round(o.activation * 1000) / 1000) + ', \nlabel: ' + label + ', \nerror: ' + Math.round(error * 1000) / 1000 + '\n- - - - - - - - - - -\n\n');
    }
    deepClone(object) {
        return JSON.parse(JSON.stringify(object));
    }
    weightFileExists(weightFile) {
        const path = 'weights/' + weightFile + '.json';
        return (0, fs_1.existsSync)(path);
    }
    readWeightFile(weightFile) {
        const path = 'weights/' + weightFile + '.json';
        const weightFileContent = (0, fs_1.readFileSync)(path, { encoding: 'utf8' });
        return JSON.parse(weightFileContent);
    }
    hasIdenticalDimensions(layers) {
        return layers.length === this.layers.length && layers.every((l, i) => {
            return l === this.layers[i];
        });
    }
    writeWeights(weightFile) {
        const path = 'weights/' + (weightFile || this.weightFile) + '.json';
        const content = { layers: this.layers, weights: this.weights };
        (0, fs_1.writeFileSync)(path, JSON.stringify(content));
    }
}
exports.CNN = CNN;
//# sourceMappingURL=cnn.js.map