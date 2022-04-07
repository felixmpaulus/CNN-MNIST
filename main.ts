import { CNN } from "./cnn";
import { generateData4Squares } from "./dataGenerator";
main();
// mainRepeatedly();

// function mainRepeatedly() {
//   for (let i = 1; i < 31; i++) {
//     console.log("\n" + i);
//     main();
//   }
// }

function main() {
  const activation = "ReLU";
  const lower = 0;
  const higher = 1;
  const biasLower = -0.5;
  const biasHigher = 0.5;
  const learningRate = 0.01;
  const momentum = 0;
  const batchSize = 1;
  const options: NNOptions = {
    activation: activation,
    weightOptions: {
      // lower, higher, biasLower, biasHigher
    },
  };

  const NN = new CNN(2, [2], 1, options);
  const data4Areas = generateData4Squares({
    numberOfDatapoints: 1000,
    xMax: 10,
    xMin: 0,
    yMax: 10,
    yMin: 0,
    divider: 5,
  });
  const training = data4Areas.slice(0, 799);
  const validation = data4Areas.slice(800, 999);

  train(NN, data4Areas, 50000, learningRate, momentum, batchSize);
  detect(NN, data4Areas);
}

function train(
  network: any,
  data: any[],
  epochs: number,
  learningRate: number,
  monentum: number,
  batchSize: number,
  weightFile?: string
) {
  const trainingData = getBatches(
    shuffle(Array(epochs).fill(data).flat()),
    batchSize
  );

  trainingData.forEach((batch: Batch) => {
    network.train(batch, learningRate, monentum);
  });

  // console.log('final weights: ')
  // console.log(JSON.stringify(network.beautifyWeights(network.weights)))
  console.log("Errors: ");
  console.log(
    network.errors.map((e: number) => e.toString().replace(".", ",")).join("\n")
  );
}

function detect(network: any, data: any[]) {
  data.forEach(({ input, label }) => {
    const output = network.detect(input);
    const guess = Math.round(output[0].activation * 100) / 100;
    const err = Math.round(Math.abs(label - guess) * 100) / 100;
    console.log(err);
    // console.log('guess: ' + guess.toFixed(2) + '      label: ' + label[0].toFixed(2) + '      error: ' + err.toFixed(2))
  });
}

function shuffle(array: any[]) {
  let currentIndex = array.length,
    randomIndex;
  while (currentIndex != 0) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex],
      array[currentIndex],
    ];
  }
  return array;
}

function getBatches(array: any[], size: number) {
  const arrayToReturn = [];
  for (let i = 0; i < array.length; i += size) {
    const chunk = array.slice(i, i + size);
    arrayToReturn.push(chunk);
  }
  return arrayToReturn;
}
