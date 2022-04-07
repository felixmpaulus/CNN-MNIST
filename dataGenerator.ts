export function generateDataXOR() {
  const dataXOR = [
    { input: [1, 1], label: [0] },
    { input: [0, 1], label: [1] },
    { input: [1, 0], label: [1] },
    { input: [0, 0], label: [0] },
  ];
  return dataXOR;
}

export function generateData4Squares(options: any) {
  const { numberOfDatapoints, xMax, xMin, yMax, yMin, divider } = options;
  const x = [
    [xMin, divider],
    [divider, xMax],
  ];
  const y = [
    [yMin, divider],
    [divider, yMax],
  ];
  const data = [];

  for (let i = 0; i < numberOfDatapoints; i++) {
    const x = Math.random() * (xMax - xMin) + xMin;
    const y = Math.random() * (yMax - yMin) + yMin;
    const label =
      (x < divider && y < divider) || (x >= divider && y >= divider) ? 1 : 0;
    data.push({ input: [x, y], label: [label] });
  }
  return data;
}
