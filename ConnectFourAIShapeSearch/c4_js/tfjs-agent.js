/**
 * TensorFlow.js policy+value agent for Connect Four Shape Search.
 * Shared trunk (policy layer sizes without last), dual heads:
 *   policy: 7 linear logits (same trainOnAction MSE as Backprop)
 *   value: 1 sigmoid = P(red wins) for search leaf eval
 * Falls back to hand-rolled NeuralNet + tiny value net when `tf` is missing.
 */
(function (global) {
  "use strict";

  var C = global.C4_CONSTANTS;
  var NN = global.C4_NN;
  var G = global.C4_GAME;
  var DEFAULT_LAYER_SIZES = (C && C.NN_POLICY_LAYER_SIZES && C.NN_POLICY_LAYER_SIZES[2]) ||
    [78, 64, 78, 7];
  /* Adam default; old SGD used 0.03 (too large for Adam). */
  var LEARNING_RATE = 0.001;
  var REWARD_WIN = (NN && NN.REWARD_WIN) || 1;
  var REWARD_LOSS = (NN && NN.REWARD_LOSS) || -4;
  var X = C.X;

  function hasTf() {
    return !!(global.tf && global.tf.layers && global.tf.train && global.tf.input);
  }

  function resolveAdamLearningRate(saved) {
    if (typeof saved !== "number") {
      return LEARNING_RATE;
    }
    /* Migrate SGD-era checkpoints that stored ~0.03. */
    if (saved >= 0.01) {
      return LEARNING_RATE;
    }
    return saved;
  }

  function createAdamOptimizer(learningRate) {
    return global.tf.train.adam(learningRate);
  }

  function sigmoid(x) {
    x = x < -20 ? -20 : x > 20 ? 20 : x;
    return 1 / (1 + Math.exp(-x));
  }

  function redWinTarget(winner) {
    if (winner === X) {
      return 1;
    }
    if (winner === C.O) {
      return 0;
    }
    return 0.5;
  }

  function valueInputFromBoard(board) {
    return G.boardToInput(board, X);
  }

  function JsFallbackNet(layerSizes) {
    this.backend = "js-fallback";
    this.layerSizes = layerSizes.slice();
    this.net = new NN.NeuralNet(layerSizes);
    this.valueNet = new NN.NeuralNet([layerSizes[0], 64, 1]);
  }

  JsFallbackNet.prototype.forward = function (input) {
    return this.net.forward(input);
  };

  JsFallbackNet.prototype.getOutputs = function (input) {
    return this.net.getOutputs(input);
  };

  JsFallbackNet.prototype.getOutputsBatch = function (inputs) {
    var out = [];
    var i;
    for (i = 0; i < inputs.length; i += 1) {
      out.push(this.getOutputs(inputs[i]));
    }
    return out;
  };

  JsFallbackNet.prototype.getValueBatch = function (inputs) {
    var out = [];
    var i;
    for (i = 0; i < inputs.length; i += 1) {
      out.push(sigmoid(this.valueNet.forward(inputs[i])[0]));
    }
    return out;
  };

  JsFallbackNet.prototype.probRedFromBoard = function (board) {
    return sigmoid(this.valueNet.forward(valueInputFromBoard(board))[0]);
  };

  JsFallbackNet.prototype.trainOnAction = function (input, actionIndex, target) {
    return this.net.trainOnAction(input, actionIndex, target);
  };

  JsFallbackNet.prototype.trainOnTrajectory = function (trajectory, reward, valueTarget) {
    var i;
    var loss = 0;
    for (i = 0; i < trajectory.length; i += 1) {
      loss += this.trainOnAction(trajectory[i].input, trajectory[i].action, reward);
      if (trajectory[i].board && typeof valueTarget === "number") {
        this.valueNet.trainOnAction(valueInputFromBoard(trajectory[i].board), 0, valueTarget);
      }
    }
    return loss;
  };

  JsFallbackNet.prototype.toJSON = function () {
    return {
      backend: this.backend,
      layerSizes: this.layerSizes,
      hasValueHead: true,
      net: this.net.toJSON(),
      valueNet: this.valueNet.toJSON()
    };
  };

  JsFallbackNet.prototype.load = function (data) {
    if (!data) {
      return;
    }
    if (data.net) {
      this.net.load(data.net);
      this.layerSizes = data.layerSizes || this.layerSizes;
    } else if (data.weights && data.biases) {
      this.net.load(data);
      this.layerSizes = data.layerSizes || this.layerSizes;
    }
    if (data.valueNet) {
      this.valueNet.load(data.valueNet);
    }
  };

  function TfjsPolicyNet(layerSizes) {
    this.backend = "tfjs";
    this.layerSizes = layerSizes.slice();
    this.learningRate = LEARNING_RATE;
    this.model = null;
    this.optimizer = null;
    this._outBuf = null;
    this._buildModel();
  }

  TfjsPolicyNet.prototype._buildModel = function () {
    var tf = global.tf;
    var sizes = this.layerSizes;
    var input = tf.input({ shape: [sizes[0]] });
    var h = input;
    var i;
    for (i = 1; i < sizes.length - 1; i += 1) {
      h = tf.layers.dense({
        units: sizes[i],
        activation: "relu",
        kernelInitializer: "heNormal",
        biasInitializer: "zeros",
        name: "shared_" + i
      }).apply(h);
    }
    var policy = tf.layers.dense({
      units: sizes[sizes.length - 1],
      activation: "linear",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
      name: "policy"
    }).apply(h);
    var value = tf.layers.dense({
      units: 1,
      activation: "sigmoid",
      kernelInitializer: "heNormal",
      biasInitializer: "zeros",
      name: "value"
    }).apply(h);
    this.model = tf.model({ inputs: input, outputs: [policy, value] });
    this.optimizer = createAdamOptimizer(this.learningRate);
    this._outBuf = new Float32Array(sizes[sizes.length - 1]);
  };

  TfjsPolicyNet.prototype._predictPair = function (xs) {
    var pred = this.model.predict(xs);
    if (Array.isArray(pred)) {
      return { policy: pred[0], value: pred[1] };
    }
    return { policy: pred, value: null };
  };

  TfjsPolicyNet.prototype.forward = function (input) {
    return this.getOutputs(input);
  };

  TfjsPolicyNet.prototype.getOutputs = function (input) {
    var tf = global.tf;
    var self = this;
    var buf = this._outBuf;
    var out = tf.tidy(function () {
      var xs = tf.tensor2d(input, [1, input.length]);
      var pair = self._predictPair(xs);
      var data = pair.policy.dataSync();
      var i;
      for (i = 0; i < data.length; i += 1) {
        buf[i] = data[i];
      }
      return buf;
    });
    return Array.prototype.slice.call(out);
  };

  TfjsPolicyNet.prototype.getOutputsBatch = function (inputs) {
    if (!inputs.length) {
      return [];
    }
    if (inputs.length === 1) {
      return [this.getOutputs(inputs[0])];
    }
    var tf = global.tf;
    var self = this;
    var n = inputs.length;
    var inputLen = inputs[0].length;
    var outWidth = this.layerSizes[this.layerSizes.length - 1];
    var flat = new Float32Array(n * inputLen);
    var i;
    var j;
    for (i = 0; i < n; i += 1) {
      var inp = inputs[i];
      var base = i * inputLen;
      for (j = 0; j < inputLen; j += 1) {
        flat[base + j] = inp[j];
      }
    }
    var data = tf.tidy(function () {
      var xs = tf.tensor2d(flat, [n, inputLen]);
      return self._predictPair(xs).policy.dataSync();
    });
    var rows = [];
    for (i = 0; i < n; i += 1) {
      var row = [];
      var off = i * outWidth;
      for (j = 0; j < outWidth; j += 1) {
        row.push(data[off + j]);
      }
      rows.push(row);
    }
    return rows;
  };

  TfjsPolicyNet.prototype.getValueBatch = function (inputs) {
    if (!inputs.length) {
      return [];
    }
    var tf = global.tf;
    var self = this;
    var n = inputs.length;
    var inputLen = inputs[0].length;
    var flat = new Float32Array(n * inputLen);
    var i;
    var j;
    for (i = 0; i < n; i += 1) {
      var inp = inputs[i];
      var base = i * inputLen;
      for (j = 0; j < inputLen; j += 1) {
        flat[base + j] = inp[j];
      }
    }
    var data = tf.tidy(function () {
      var xs = tf.tensor2d(flat, [n, inputLen]);
      var pair = self._predictPair(xs);
      if (!pair.value) {
        return null;
      }
      return pair.value.dataSync();
    });
    if (!data) {
      return this.getOutputsBatch(inputs).map(function () { return 0.5; });
    }
    var values = [];
    for (i = 0; i < n; i += 1) {
      values.push(data[i]);
    }
    return values;
  };

  TfjsPolicyNet.prototype.probRedFromBoard = function (board) {
    return this.getValueBatch([valueInputFromBoard(board)])[0];
  };

  TfjsPolicyNet.prototype.trainOnAction = function (input, actionIndex, target) {
    return this.trainOnTrajectory([{ input: input, action: actionIndex }], target, 0.5);
  };

  /**
   * One optimizer step: policy MSE on played actions + value MSE toward P(red wins).
   */
  TfjsPolicyNet.prototype.trainOnTrajectory = function (trajectory, reward, valueTarget) {
    if (!trajectory.length) {
      return 0;
    }
    if (typeof valueTarget !== "number") {
      valueTarget = 0.5;
    }
    var tf = global.tf;
    var self = this;
    var optimizer = this.optimizer;
    var n = trajectory.length;
    var inputLen = trajectory[0].input.length;
    var policyFlat = new Float32Array(n * inputLen);
    var valueFlat = new Float32Array(n * inputLen);
    var actions = new Int32Array(n);
    var i;
    var j;
    for (i = 0; i < n; i += 1) {
      var inp = trajectory[i].input;
      var base = i * inputLen;
      for (j = 0; j < inputLen; j += 1) {
        policyFlat[base + j] = inp[j];
      }
      actions[i] = trajectory[i].action;
      var vin = trajectory[i].board
        ? valueInputFromBoard(trajectory[i].board)
        : inp;
      for (j = 0; j < inputLen; j += 1) {
        valueFlat[base + j] = vin[j];
      }
    }
    var lossValue = 0;
    var actionCount = this.layerSizes[this.layerSizes.length - 1];
    optimizer.minimize(function () {
      var pxs = tf.tensor2d(policyFlat, [n, inputLen]);
      var vxs = tf.tensor2d(valueFlat, [n, inputLen]);
      var pPair = self._predictPair(pxs);
      var vPair = self._predictPair(vxs);
      var mask = tf.oneHot(tf.tensor1d(actions, "int32"), actionCount);
      var actionPred = pPair.policy.mul(mask).sum(1);
      var policyLoss = actionPred.sub(tf.fill([n], reward)).square().mean();
      var valueLoss = vPair.value.reshape([n]).sub(tf.fill([n], valueTarget)).square().mean();
      var loss = policyLoss.add(valueLoss);
      lossValue = loss.dataSync()[0];
      return loss;
    });
    return lossValue;
  };

  TfjsPolicyNet.prototype.toJSON = function () {
    var weights = this.model.getWeights();
    var serialized = [];
    var i;
    for (i = 0; i < weights.length; i += 1) {
      serialized.push({
        shape: weights[i].shape.slice(),
        data: Array.from(weights[i].dataSync())
      });
    }
    return {
      backend: this.backend,
      layerSizes: this.layerSizes,
      learningRate: this.learningRate,
      hasValueHead: true,
      weights: serialized
    };
  };

  TfjsPolicyNet.prototype.load = function (data) {
    if (!data || !data.weights || !data.weights.length) {
      return;
    }
    if (!data.hasValueHead) {
      /* Old single-head checkpoints are incompatible; keep fresh dual-head weights. */
      return;
    }
    var tf = global.tf;
    var tensors = [];
    var i;
    try {
      for (i = 0; i < data.weights.length; i += 1) {
        var w = data.weights[i];
        tensors.push(tf.tensor(w.data, w.shape));
      }
      this.model.setWeights(tensors);
    } catch (err) {
      /* Shape mismatch: leave randomly initialized dual-head. */
    }
    for (i = 0; i < tensors.length; i += 1) {
      tensors[i].dispose();
    }
    this.learningRate = resolveAdamLearningRate(data.learningRate);
    this.optimizer = createAdamOptimizer(this.learningRate);
  };

  function createPolicyNet(layerSizes) {
    if (hasTf()) {
      return new TfjsPolicyNet(layerSizes);
    }
    return new JsFallbackNet(layerSizes);
  }

  function TfjsPolicyAgent(layerSizes) {
    this.layerSizes = (layerSizes || DEFAULT_LAYER_SIZES).slice();
    this.net = createPolicyNet(this.layerSizes);
    this.trainSteps = 0;
  }

  TfjsPolicyAgent.prototype.backend = function () {
    var b = this.net.backend;
    if (b === "tfjs" && hasTf() && global.tf.getBackend) {
      return "tfjs:" + global.tf.getBackend();
    }
    return b;
  };

  /* chooseMove / chooseMoveTimed / chooseMoveMinimax are attached in engine.js */

  TfjsPolicyAgent.prototype.learnFromTrajectory = function (trajectory, mark, winner) {
    if (!trajectory.length) {
      return;
    }
    var reward = 0;
    if (winner === mark) {
      reward = REWARD_WIN;
    } else if (winner && winner !== mark) {
      reward = REWARD_LOSS;
    }
    var steps = G.expandTrajectoryWithMirrors(trajectory);
    this.net.trainOnTrajectory(steps, reward, redWinTarget(winner));
    this.trainSteps += steps.length;
  };

  TfjsPolicyAgent.prototype.buildStep = function (board, player, action) {
    return {
      input: G.boardToInput(board, player),
      action: action,
      board: G.cloneBoard(board),
      player: player
    };
  };

  TfjsPolicyAgent.prototype.toDict = function () {
    return {
      net: this.net.toJSON(),
      trainSteps: this.trainSteps,
      layerSizes: this.layerSizes
    };
  };

  TfjsPolicyAgent.fromDict = function (data) {
    var sizes = (data && data.layerSizes) || DEFAULT_LAYER_SIZES;
    var agent = new TfjsPolicyAgent(sizes);
    if (data) {
      if (data.net) {
        /* Backend Keras weights are not used; keep a fresh TF.js net if seen. */
        if (data.net.backend === "tf-keras") {
          agent.trainSteps = data.trainSteps || 0;
          return agent;
        }
        if (data.net.backend === "tfjs" && hasTf()) {
          if (agent.net.backend !== "tfjs") {
            agent.net = new TfjsPolicyNet(sizes);
          }
          agent.net.load(data.net);
        } else if (data.net.backend === "tfjs" && !hasTf()) {
          agent.net = new JsFallbackNet(sizes);
        } else {
          if (agent.net.backend !== "js-fallback") {
            agent.net = new JsFallbackNet(sizes);
          }
          agent.net.load(data.net);
        }
      }
      agent.trainSteps = data.trainSteps || 0;
    }
    return agent;
  };

  /* --- Value-only TF.js net (matches Backprop value 2L: 77 -> ... -> 1) --- */
  var DEFAULT_VALUE_LAYER_SIZES = (C && C.NN_VALUE_LAYER_SIZES && C.NN_VALUE_LAYER_SIZES[2]) ||
    [77, 64, 77, 1];
  /* Blend of Monte Carlo outcome and bootstrap V(next learner board). */
  var VALUE_TD_MIX = 0.7;
  var VALUE_TD_GAMMA = 1.0;

  function redInputFromBoard(board) {
    return G.boardToRedInput(board);
  }

  function clamp01(x) {
    return x < 0 ? 0 : x > 1 ? 1 : x;
  }

  function normalizeValueTargets(n, valueTargets) {
    var targets = new Float32Array(n);
    var i;
    if (typeof valueTargets === "number") {
      for (i = 0; i < n; i += 1) {
        targets[i] = valueTargets;
      }
      return targets;
    }
    if (!valueTargets || valueTargets.length !== n) {
      for (i = 0; i < n; i += 1) {
        targets[i] = 0.5;
      }
      return targets;
    }
    for (i = 0; i < n; i += 1) {
      targets[i] = valueTargets[i];
    }
    return targets;
  }

  /**
   * Per-step targets: mix terminal P(red wins) with V(next trajectory board).
   * Mirror pairs keep the same target (left-right symmetry).
   */
  function buildTdValuePairs(net, trajectory, winner) {
    var outcome = redWinTarget(winner);
    var pairs = [];
    var i;
    for (i = 0; i < trajectory.length; i += 1) {
      if (!trajectory[i] || !trajectory[i].board) {
        continue;
      }
      var target = outcome;
      if (i < trajectory.length - 1 && trajectory[i + 1] && trajectory[i + 1].board) {
        var boot = net.probRedFromBoard(trajectory[i + 1].board);
        target = (1 - VALUE_TD_MIX) * outcome + VALUE_TD_MIX * VALUE_TD_GAMMA * boot;
        target = clamp01(target);
      }
      pairs.push({ board: trajectory[i].board, target: target });
      pairs.push({ board: G.mirrorBoard(trajectory[i].board), target: target });
    }
    return pairs;
  }

  function JsFallbackValueNet(layerSizes) {
    this.backend = "js-fallback";
    this.layerSizes = layerSizes.slice();
    this.net = new NN.NeuralNet(layerSizes);
  }

  JsFallbackValueNet.prototype.getValueBatch = function (inputs) {
    var out = [];
    var i;
    for (i = 0; i < inputs.length; i += 1) {
      out.push(sigmoid(this.net.forward(inputs[i])[0]));
    }
    return out;
  };

  JsFallbackValueNet.prototype.probRedFromBoard = function (board) {
    return sigmoid(this.net.forward(redInputFromBoard(board))[0]);
  };

  JsFallbackValueNet.prototype.trainOnBoards = function (boards, valueTargets) {
    var targets = normalizeValueTargets(boards.length, valueTargets);
    var i;
    var loss = 0;
    for (i = 0; i < boards.length; i += 1) {
      loss += this.net.trainOnAction(redInputFromBoard(boards[i]), 0, targets[i]);
    }
    return loss;
  };

  JsFallbackValueNet.prototype.toJSON = function () {
    return {
      backend: this.backend,
      layerSizes: this.layerSizes,
      net: this.net.toJSON()
    };
  };

  JsFallbackValueNet.prototype.load = function (data) {
    if (!data) {
      return;
    }
    if (data.net) {
      this.net.load(data.net);
      this.layerSizes = data.layerSizes || this.layerSizes;
    } else if (data.weights) {
      this.net.load(data);
    }
  };

  function TfjsValueNet(layerSizes) {
    this.backend = "tfjs";
    this.layerSizes = layerSizes.slice();
    this.learningRate = LEARNING_RATE;
    this.model = null;
    this.optimizer = null;
    this._buildModel();
  }

  TfjsValueNet.prototype._buildModel = function () {
    var tf = global.tf;
    var sizes = this.layerSizes;
    var model = tf.sequential();
    var i;
    for (i = 1; i < sizes.length; i += 1) {
      var isLast = i === sizes.length - 1;
      var cfg = {
        units: sizes[i],
        activation: isLast ? "sigmoid" : "relu",
        kernelInitializer: "heNormal",
        biasInitializer: "zeros"
      };
      if (i === 1) {
        cfg.inputShape = [sizes[0]];
      }
      model.add(tf.layers.dense(cfg));
    }
    this.model = model;
    this.optimizer = createAdamOptimizer(this.learningRate);
  };

  TfjsValueNet.prototype.getValueBatch = function (inputs) {
    if (!inputs.length) {
      return [];
    }
    var tf = global.tf;
    var model = this.model;
    var n = inputs.length;
    var inputLen = inputs[0].length;
    var flat = new Float32Array(n * inputLen);
    var i;
    var j;
    for (i = 0; i < n; i += 1) {
      var inp = inputs[i];
      var base = i * inputLen;
      for (j = 0; j < inputLen; j += 1) {
        flat[base + j] = inp[j];
      }
    }
    var data = tf.tidy(function () {
      var xs = tf.tensor2d(flat, [n, inputLen]);
      return model.predict(xs).dataSync();
    });
    var values = [];
    for (i = 0; i < n; i += 1) {
      values.push(data[i]);
    }
    return values;
  };

  TfjsValueNet.prototype.probRedFromBoard = function (board) {
    return this.getValueBatch([redInputFromBoard(board)])[0];
  };

  TfjsValueNet.prototype.trainOnBoards = function (boards, valueTargets) {
    if (!boards.length) {
      return 0;
    }
    var targets = normalizeValueTargets(boards.length, valueTargets);
    var tf = global.tf;
    var model = this.model;
    var optimizer = this.optimizer;
    var n = boards.length;
    var sample = redInputFromBoard(boards[0]);
    var inputLen = sample.length;
    var flat = new Float32Array(n * inputLen);
    var i;
    var j;
    for (i = 0; i < n; i += 1) {
      var inp = redInputFromBoard(boards[i]);
      var base = i * inputLen;
      for (j = 0; j < inputLen; j += 1) {
        flat[base + j] = inp[j];
      }
    }
    var lossValue = 0;
    optimizer.minimize(function () {
      var xs = tf.tensor2d(flat, [n, inputLen]);
      var pred = model.predict(xs).reshape([n]);
      var tgt = tf.tensor1d(targets);
      var loss = pred.sub(tgt).square().mean();
      lossValue = loss.dataSync()[0];
      return loss;
    });
    return lossValue;
  };

  TfjsValueNet.prototype.toJSON = function () {
    var weights = this.model.getWeights();
    var serialized = [];
    var i;
    for (i = 0; i < weights.length; i += 1) {
      serialized.push({
        shape: weights[i].shape.slice(),
        data: Array.from(weights[i].dataSync())
      });
    }
    return {
      backend: this.backend,
      layerSizes: this.layerSizes,
      learningRate: this.learningRate,
      weights: serialized
    };
  };

  TfjsValueNet.prototype.load = function (data) {
    if (!data || !data.weights || !data.weights.length) {
      return;
    }
    var tf = global.tf;
    var tensors = [];
    var i;
    try {
      for (i = 0; i < data.weights.length; i += 1) {
        var w = data.weights[i];
        tensors.push(tf.tensor(w.data, w.shape));
      }
      this.model.setWeights(tensors);
    } catch (err) {
      /* leave fresh weights */
    }
    for (i = 0; i < tensors.length; i += 1) {
      tensors[i].dispose();
    }
    this.learningRate = resolveAdamLearningRate(data.learningRate);
    this.optimizer = createAdamOptimizer(this.learningRate);
  };

  function createValueNet(layerSizes) {
    if (hasTf()) {
      return new TfjsValueNet(layerSizes);
    }
    return new JsFallbackValueNet(layerSizes);
  }

  function TfjsValueAgent(layerSizes) {
    this.layerSizes = (layerSizes || DEFAULT_VALUE_LAYER_SIZES).slice();
    this.net = createValueNet(this.layerSizes);
    this.trainSteps = 0;
  }

  TfjsValueAgent.prototype.backend = function () {
    var b = this.net.backend;
    if (b === "tfjs" && hasTf() && global.tf.getBackend) {
      return "tfjs:" + global.tf.getBackend();
    }
    return b;
  };

  TfjsValueAgent.prototype.probRedWins = function (board) {
    return this.net.probRedFromBoard(board);
  };

  /* chooseMove / timed / minimax attached in engine.js */

  TfjsValueAgent.prototype.learnFromTrajectory = function (trajectory, mark, winner) {
    if (!trajectory.length) {
      return;
    }
    var pairs = buildTdValuePairs(this.net, trajectory, winner);
    if (!pairs.length) {
      return;
    }
    var boards = [];
    var targets = [];
    var i;
    for (i = 0; i < pairs.length; i += 1) {
      boards.push(pairs[i].board);
      targets.push(pairs[i].target);
    }
    this.net.trainOnBoards(boards, targets);
    this.trainSteps += boards.length;
  };

  TfjsValueAgent.prototype.buildStep = function (board, player, action) {
    return { board: G.cloneBoard(board) };
  };

  TfjsValueAgent.prototype.toDict = function () {
    return {
      net: this.net.toJSON(),
      trainSteps: this.trainSteps,
      layerSizes: this.layerSizes
    };
  };

  TfjsValueAgent.fromDict = function (data) {
    var sizes = (data && data.layerSizes) || DEFAULT_VALUE_LAYER_SIZES;
    var agent = new TfjsValueAgent(sizes);
    if (data) {
      if (data.net) {
        if (data.net.backend === "tf-keras") {
          agent.trainSteps = data.trainSteps || 0;
          return agent;
        }
        if (data.net.backend === "tfjs" && hasTf()) {
          if (agent.net.backend !== "tfjs") {
            agent.net = new TfjsValueNet(sizes);
          }
          agent.net.load(data.net);
        } else if (data.net.backend === "tfjs" && !hasTf()) {
          agent.net = new JsFallbackValueNet(sizes);
        } else {
          if (agent.net.backend !== "js-fallback") {
            agent.net = new JsFallbackValueNet(sizes);
          }
          agent.net.load(data.net);
        }
      }
      agent.trainSteps = data.trainSteps || 0;
    }
    return agent;
  };

  global.C4_TFJS = {
    TfjsPolicyAgent: TfjsPolicyAgent,
    TfjsValueAgent: TfjsValueAgent,
    hasTf: hasTf,
    LEARNING_RATE: LEARNING_RATE,
    resolveAdamLearningRate: resolveAdamLearningRate,
    DEFAULT_LAYER_SIZES: DEFAULT_LAYER_SIZES,
    DEFAULT_VALUE_LAYER_SIZES: DEFAULT_VALUE_LAYER_SIZES,
    VALUE_TD_MIX: VALUE_TD_MIX,
    VALUE_TD_GAMMA: VALUE_TD_GAMMA,
    buildTdValuePairs: buildTdValuePairs
  };
})(typeof window !== "undefined" ? window : global);
