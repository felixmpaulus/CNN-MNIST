"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activationFunctions = void 0;
exports.activationFunctions = {
    'ReLU': { primitive: ReLU, derivative: derivedReLU }
};
function ReLU(value) {
    return value > 0 ? value : 0;
}
function derivedReLU(value) {
    return value > 0 ? 1 : 0;
}
//# sourceMappingURL=activation.js.map