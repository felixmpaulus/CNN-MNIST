export const activationFunctions: {
    [activation in ActivationType]:
    {
        'primitive': (value: number) => number, 'derivative': (value: number) => number,
    } } = {
    'ReLU': {
        primitive: (value: number): number => { return value > 0 ? value : 0 },
        derivative: (value: number): number => { return value > 0 ? 1 : 0 }
    },
    'leakyReLU': {
        primitive: (value: number) => { return value > 0 ? value : (0.001 * value) },
        derivative: (value: number) => { return value > 0 ? 1 : (-0.001) }
    },
    'sigmoid': {
        primitive: (value: number) => { return 1 / (1 + Math.exp(-value)) },
        derivative: (value: number) => { return (1 / (1 + Math.exp(-value))) * (1 - (1 / (1 + Math.exp(-value)))) }
    }
}