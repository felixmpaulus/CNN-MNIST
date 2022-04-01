"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activationFunctions = void 0;
exports.activationFunctions = {
    'ReLU': {
        primitive: (value) => { return value > 0 ? value : 0; },
        derivative: (value) => { return value > 0 ? 1 : 0; }
    },
    'leakyReLU': {
        primitive: (value) => { return value > 0 ? value : (0.001 * value); },
        derivative: (value) => { return value > 0 ? 1 : (-0.001); }
    },
    'sigmoid': {
        primitive: (value) => { return 1 / (1 + Math.exp(-value)); },
        derivative: (value) => { return (1 / (1 + Math.exp(-value))) * (1 - (1 / (1 + Math.exp(-value)))); }
    }
};
//# sourceMappingURL=activation.js.map