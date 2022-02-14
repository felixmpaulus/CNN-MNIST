type Label = number[]
type Layer = Neuron[]
type Neuron = {
    value: number
}
type Activation = {
    root: (v: number) => number
    derivative: (v: number) => number
}
type ActivationType = 'ReLU'
type Weight = number
