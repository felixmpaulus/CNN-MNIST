type Label = number[]
type Layer = Neuron[]
type Neuron = {
    value: number
}
type Activation = (v: number) => number
type ActivationType = 'ReLU'
type Weight = number
