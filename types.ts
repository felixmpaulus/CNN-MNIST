type Label = number[]
type Layer = Neuron[]
type InputLayer = number[]
type Neuron = {
    dotProduct?: number
    activation: number
    error?: number
}
type Activation = {
    primitive: (v: number) => number
    derivative: (v: number) => number
}
type ActivationType = 'ReLU' | 'leakyReLU' | 'sigmoid'
type Weight = number
type Bias = number
type Weights = { weights: Weight[], bias: Bias }[][]
type NNOptions = {
    activation: ActivationType,
    weightOptions: WeightOptions,
}
type WeightOptions = {
    weightFile?: string,
    fixedWeights?: Weights,
    lowerLimit?: Weight,
    higherLimit?: Weight,
}
