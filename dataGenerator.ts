// const numberOfDatapoints = 400
// const xMax = 10
// const xMin = 0
// const yMax = 10
// const yMin = 0
// const divider = 5

// 
// 0 0 1 1
// 0 0 1 1
// 1 1 0 0
// 1 1 0 0
// 

export default function generateData(options: any) {
    const { numberOfDatapoints, xMax, xMin, yMax, yMin, divider } = options
    const x = [[xMin, divider], [divider, xMax]]
    const y = [[yMin, divider], [divider, yMax]]
    const data = []

    for (let i = 0; i < numberOfDatapoints; i++) {
        const x = Math.random() * (xMax - xMin) + xMin
        const y = Math.random() * (yMax - yMin) + yMin
        const label = ((x < divider && y < divider) || (x >= divider && y >= divider)) ? 1 : 0
        data.push({ input: [x, y], label: [label] })
    }
    return data
}