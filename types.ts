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
type ActivationType = 'ReLU'
type Weight = number
type Weights = Weight[][][]
type NNOptions = {
    activation: ActivationType,
    weightOptions: WeightOptions,
}
type WeightOptions = {
    weightFile?: string,
    fixedWeights?: Weight[][][],
    lowerLimit?: Weight,
    higherLimit?: Weight,
}
