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
