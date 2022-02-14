type Label = number[]
type Layer = Neuron[]
type Neuron = {
    dotProduct: number
    activation: number
    error: number
}
type Activation = {
    root: (v: number) => number
    derivative: (v: number) => number
}
type ActivationType = 'ReLU'
type Weight = number
