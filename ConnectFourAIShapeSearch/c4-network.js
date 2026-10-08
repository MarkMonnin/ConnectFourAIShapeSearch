(function (global) {
  "use strict";

  var NN_LAYER_SIZES = [43, 64, 43, 7];
  var NN_LEARNING_RATE = 0.03;

  function relu(x) {
    return x > 0 ? x : 0;
  }

  function reluDerivFromPre(pre) {
    return pre > 0 ? 1 : 0;
  }

  function randScale(fanIn) {
    return Math.sqrt(2 / fanIn);
  }

  function randomMatrix(rows, cols, scale) {
    var m = [];
    for (var r = 0; r < rows; r += 1) {
      var row = [];
      for (var c = 0; c < cols; c += 1) {
        row.push((Math.random() * 2 - 1) * scale);
      }
      m.push(row);
    }
    return m;
  }

  function zeroVector(size) {
    var v = [];
    for (var i = 0; i < size; i += 1) {
      v.push(0);
    }
    return v;
  }

  function matVecMul(matrix, vector) {
    var out = [];
    for (var r = 0; r < matrix.length; r += 1) {
      var sum = 0;
      for (var c = 0; c < vector.length; c += 1) {
        sum += matrix[r][c] * vector[c];
      }
      out.push(sum);
    }
    return out;
  }

  function vecAdd(a, b) {
    var out = [];
    for (var i = 0; i < a.length; i += 1) {
      out.push(a[i] + b[i]);
    }
    return out;
  }

  function NeuralNet(layerSizes, learningRate) {
    this.layerSizes = layerSizes.slice();
    this.learningRate = typeof learningRate === "number" ? learningRate : NN_LEARNING_RATE;
    this.weights = [];
    this.biases = [];
    for (var i = 0; i < layerSizes.length - 1; i += 1) {
      var fanIn = layerSizes[i];
      var fanOut = layerSizes[i + 1];
      this.weights.push(randomMatrix(fanOut, fanIn, randScale(fanIn)));
      this.biases.push(zeroVector(fanOut));
    }
    this.activations = [];
    this.preActivations = [];
  }

  NeuralNet.prototype.forward = function (input) {
    this.activations = [input.slice()];
    this.preActivations = [];
    var activation = input;
    for (var layer = 0; layer < this.weights.length; layer += 1) {
      var pre = vecAdd(matVecMul(this.weights[layer], activation), this.biases[layer]);
      this.preActivations.push(pre);
      if (layer < this.weights.length - 1) {
        activation = pre.map(relu);
      } else {
        activation = pre.slice();
      }
      this.activations.push(activation);
    }
    return activation;
  };

  NeuralNet.prototype.trainOnAction = function (input, actionIndex, target) {
    var output = this.forward(input);
    var gradOutput = zeroVector(output.length);
    var error = output[actionIndex] - target;
    gradOutput[actionIndex] = error;

    var grad = gradOutput;
    for (var layer = this.weights.length - 1; layer >= 0; layer -= 1) {
      var prevActivation = this.activations[layer];
      var weightGrad = [];
      for (var r = 0; r < this.weights[layer].length; r += 1) {
        var rowGrad = [];
        for (var c = 0; c < prevActivation.length; c += 1) {
          rowGrad.push(grad[r] * prevActivation[c]);
        }
        weightGrad.push(rowGrad);
      }

      for (var wr = 0; wr < this.weights[layer].length; wr += 1) {
        for (var wc = 0; wc < this.weights[layer][wr].length; wc += 1) {
          this.weights[layer][wr][wc] -= this.learningRate * weightGrad[wr][wc];
        }
        this.biases[layer][wr] -= this.learningRate * grad[wr];
      }

      if (layer === 0) {
        break;
      }

      var nextGrad = zeroVector(prevActivation.length);
      for (var j = 0; j < prevActivation.length; j += 1) {
        var sum = 0;
        for (var k = 0; k < grad.length; k += 1) {
          sum += this.weights[layer][k][j] * grad[k];
        }
        nextGrad[j] = sum * reluDerivFromPre(this.preActivations[layer - 1][j]);
      }
      grad = nextGrad;
    }
    return error * error;
  };

  NeuralNet.prototype.getOutputs = function (input) {
    return this.forward(input).slice();
  };

  NeuralNet.prototype.weightAbsRange = function () {
    var min = Infinity;
    var max = 0;
    for (var l = 0; l < this.weights.length; l += 1) {
      for (var r = 0; r < this.weights[l].length; r += 1) {
        for (var c = 0; c < this.weights[l][r].length; c += 1) {
          var abs = Math.abs(this.weights[l][r][c]);
          if (abs < min) {
            min = abs;
          }
          if (abs > max) {
            max = abs;
          }
        }
      }
    }
    if (min === Infinity) {
      min = 0;
    }
    if (max <= min) {
      max = min + 0.001;
    }
    return { min: min, max: max };
  };

  NeuralNet.prototype.toJSON = function () {
    return {
      layerSizes: this.layerSizes,
      weights: this.weights,
      biases: this.biases
    };
  };

  NeuralNet.prototype.load = function (data) {
    if (!data) {
      return;
    }
    this.layerSizes = data.layerSizes || NN_LAYER_SIZES.slice();
    this.weights = data.weights || this.weights;
    this.biases = data.biases || this.biases;
  };

  function boardToInput(board, player) {
    if (global.C4_GAME && global.C4_GAME.boardToInput) {
      return global.C4_GAME.boardToInput(board, player);
    }
    throw new Error("C4_GAME.boardToInput required");
  }

  function pickMoveFromOutputs(outputs, legalMoves, explore) {
    if (explore && Math.random() < 0.1) {
      return legalMoves[Math.floor(Math.random() * legalMoves.length)];
    }
    var best = -Infinity;
    var picks = [];
    for (var i = 0; i < legalMoves.length; i += 1) {
      var move = legalMoves[i];
      if (outputs[move] > best) {
        best = outputs[move];
        picks = [move];
      } else if (outputs[move] === best) {
        picks.push(move);
      }
    }
    return picks[Math.floor(Math.random() * picks.length)];
  }

  global.C4_NN = {
    NN_LAYER_SIZES: NN_LAYER_SIZES,
    REWARD_WIN: 1,
    REWARD_LOSS: -4,
    NeuralNet: NeuralNet,
    boardToInput: boardToInput,
    pickMoveFromOutputs: pickMoveFromOutputs
  };
})(window);
