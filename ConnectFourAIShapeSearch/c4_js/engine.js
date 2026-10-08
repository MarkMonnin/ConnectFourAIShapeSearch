(function (global) {
  "use strict";

  var C = global.C4_CONSTANTS;
  var G = global.C4_GAME;
  var M = global.C4_MINIMAX;
  var NN = global.C4_NN;
  var OPP = global.C4_OPPONENTS;
  var TFJS = global.C4_TFJS;

  var X = C.X;
  var O = C.O;
  var TRAINING_ALGO_IDS = C.TRAINING_ALGO_IDS;
  var VALUE_AGENT_IDS = C.VALUE_AGENT_IDS;
  var ALL_ALGO_IDS = C.ALL_ALGO_IDS;
  var ALGO_NAMES = C.ALGO_NAMES;

  /* Softmax explore for policy nets; ε decays with games played (see learnings nn-explore). */
  var NN_EXPLORE_RATE_START = 0.25;
  var NN_EXPLORE_RATE_MIN = 0.05;
  var NN_EXPLORE_DECAY_GAMES = 3000;
  var NN_EXPLORE_TEMPERATURE = 1.0;
  var NN_LEARNING_RATE = 0.03;
  var VALUE_NET_LAYER_SIZES = (C.NN_VALUE_LAYER_SIZES && C.NN_VALUE_LAYER_SIZES[2]) ||
    [42, 64, 42, 1];
  var NN_POLICY_LAYER_SIZES = C.NN_POLICY_LAYER_SIZES || { 2: NN.NN_LAYER_SIZES };
  var NN_VALUE_LAYER_SIZES = C.NN_VALUE_LAYER_SIZES || { 2: VALUE_NET_LAYER_SIZES };
  var VALUE_NET_LEARNING_RATE = 0.03;
  var VALUE_MINIMAX_DEPTH = C.NN_TRAIN_MINIMAX_DEPTH || 1;
  var NN_TOURNAMENT_DEPTH = C.NN_TOURNAMENT_DEPTH || 1;
  var NEAT_GA_LAYER_SIZES = [C.POLICY_INPUT_DIM || 78, 16, 7];
  var NEAT_VALUE_GA_LAYER_SIZES = [C.VALUE_INPUT_DIM || 77, 16, 1];
  var REWARD_WIN = 1;
  var REWARD_LOSS = -4;
  var LOSS_REPLAY_COUNT = OPP.LOSS_REPLAY_COUNT;

  function displayName(algoId) {
    return ALGO_NAMES[algoId] || algoId;
  }

  function nnVariantDepth(algoId) {
    var m = /^(nn_value|reinforce_value|reinforce|nn)(\d)$/.exec(algoId);
    return m ? parseInt(m[2], 10) : null;
  }

  function isTfjsAlgo(algoId) {
    return algoId === "tfjs" || algoId === "tfjs_value" || algoId === "tfjs_value5";
  }

  function isTrainableNn(algoId) {
    return nnVariantDepth(algoId) !== null || isTfjsAlgo(algoId);
  }

  function isNnLearner(algoId) {
    return isTrainableNn(algoId) || algoId === "neat" || algoId === "neat_value";
  }

  function migrateLegacyAgents(data) {
    if (!data) {
      return data;
    }
    var pairs = [
      ["nn", "nn2"],
      ["nn_value", "nn_value2"],
      ["reinforce", "reinforce2"],
      ["reinforce_value", "reinforce_value2"]
    ];
    var i;
    for (i = 0; i < pairs.length; i += 1) {
      if (data[pairs[i][0]] && !data[pairs[i][1]]) {
        data[pairs[i][1]] = data[pairs[i][0]];
      }
    }
    return data;
  }

  function migrateLegacyStates(data) {
    if (!data) {
      return data;
    }
    var pairs = [
      ["nn", "nn2"],
      ["nn_value", "nn_value2"],
      ["reinforce", "reinforce2"],
      ["reinforce_value", "reinforce_value2"]
    ];
    var i;
    for (i = 0; i < pairs.length; i += 1) {
      if (data[pairs[i][0]] && !data[pairs[i][1]]) {
        data[pairs[i][1]] = data[pairs[i][0]];
      }
    }
    return data;
  }

  var NN_VARIANT_DEPTHS = [2, 3];

  var execTrace = [];
  var MAX_EXEC_TRACE = 40;

  function tracePush(label, detail) {
    execTrace.push({
      label: label,
      detail: detail || null,
      at: Date.now()
    });
    if (execTrace.length > MAX_EXEC_TRACE) {
      execTrace.shift();
    }
  }

  function tracePop() {
    if (execTrace.length) {
      execTrace.pop();
    }
  }

  function tracedCall(label, detail, fn) {
    tracePush(label, detail);
    try {
      return fn();
    } finally {
      tracePop();
    }
  }

  function formatErrorReport(err, app) {
    var lines = [];
    var msg = err && err.message ? err.message : String(err);
    lines.push(msg);
    if (app) {
      lines.push("");
      lines.push("Runtime context:");
      lines.push("  phase: " + (app.phase || "?"));
      try {
        lines.push("  activity: " + app.activityLabel());
        if (app.training) {
          lines.push("  current algo: " + displayName(app.training.currentAlgoId()));
          if (app.training._handlingExploreLoss) {
            lines.push("  handlingExploreLoss: true");
          }
        }
      } catch (ctxErr) {
        lines.push("  (context unavailable: " + (ctxErr.message || ctxErr) + ")");
      }
    }
    if (execTrace.length) {
      lines.push("");
      lines.push("Execution trace (last entry = innermost):");
      var i;
      for (i = 0; i < execTrace.length; i += 1) {
        var t = execTrace[i];
        var row = "  " + (i + 1) + ". " + t.label;
        if (t.detail) {
          try {
            row += " " + JSON.stringify(t.detail);
          } catch (jsonErr) {
            row += " [detail]";
          }
        }
        lines.push(row);
      }
      lines.push("  >> last: " + execTrace[execTrace.length - 1].label);
    }
    if (err && err.stack) {
      lines.push("");
      lines.push("Stack trace:");
      var stackLines = String(err.stack).split("\n");
      var maxStack = msg.indexOf("Maximum call stack") >= 0 ? 20 : 14;
      for (var s = 0; s < stackLines.length && s < maxStack; s += 1) {
        lines.push(stackLines[s]);
      }
      if (stackLines.length > maxStack) {
        lines.push("  ... (" + (stackLines.length - maxStack) + " more frames)");
      }
    }
    return lines.join("\n");
  }

  function cloneBoard(board) {
    return G.cloneBoard(board);
  }

  function buildTabularStep(board, player, action) {
    return {
      stateKey: G.boardKey(board) + "|" + player,
      action: action,
      player: player,
      moves: G.legalMoves(board)
    };
  }

  function boardToInput(board, player) {
    return G.boardToInput(board, player);
  }

  function boardToRedInput(board) {
    return G.boardToRedInput(board);
  }

  function redWinTarget(winner) {
    if (winner === X) {
      return 1;
    }
    if (winner === O) {
      return 0;
    }
    return 0.5;
  }

  function utilityForPlayer(pRed, player) {
    return player === X ? pRed : 1 - pRed;
  }

  function sigmoid(x) {
    x = x < -20 ? -20 : x > 20 ? 20 : x;
    return 1 / (1 + Math.exp(-x));
  }

  function clipWeights(net) {
    var l;
    for (l = 0; l < net.weights.length; l += 1) {
      var r;
      for (r = 0; r < net.weights[l].length; r += 1) {
        var c;
        for (c = 0; c < net.weights[l][r].length; c += 1) {
          var w = net.weights[l][r][c];
          net.weights[l][r][c] = w < -8 ? -8 : w > 8 ? 8 : w;
        }
        var b = net.biases[l][r];
        net.biases[l][r] = b < -8 ? -8 : b > 8 ? 8 : b;
      }
    }
  }

  function outer(delta, prev) {
    var rows = [];
    var r;
    for (r = 0; r < delta.length; r += 1) {
      var row = [];
      var c;
      for (c = 0; c < prev.length; c += 1) {
        row.push(delta[r] * prev[c]);
      }
      rows.push(row);
    }
    return rows;
  }

  function matVecT(weights, delta) {
    var out = [];
    var j;
    for (j = 0; j < weights[0].length; j += 1) {
      var sum = 0;
      var k;
      for (k = 0; k < delta.length; k += 1) {
        sum += weights[k][j] * delta[k];
      }
      out.push(sum);
    }
    return out;
  }

  /* --- ValueNeuralNet --- */
  function ValueNeuralNet(layerSizes, learningRate) {
    this.layerSizes = layerSizes || VALUE_NET_LAYER_SIZES.slice();
    this.learningRate = typeof learningRate === "number" ? learningRate : VALUE_NET_LEARNING_RATE;
    this.weights = [];
    this.biases = [];
    this.activations = [];
    this.preActivations = [];
    var i;
    for (i = 0; i < this.layerSizes.length - 1; i += 1) {
      var fanIn = this.layerSizes[i];
      var fanOut = this.layerSizes[i + 1];
      var scale = Math.sqrt(2 / fanIn);
      var w = [];
      var o;
      for (o = 0; o < fanOut; o += 1) {
        var row = [];
        var j;
        for (j = 0; j < fanIn; j += 1) {
          row.push((Math.random() * 2 - 1) * scale);
        }
        w.push(row);
      }
      this.weights.push(w);
      var b = [];
      for (o = 0; o < fanOut; o += 1) {
        b.push(0);
      }
      this.biases.push(b);
    }
  }

  ValueNeuralNet.prototype.forward = function (inp) {
    this.activations = [inp.slice()];
    this.preActivations = [];
    var act = inp;
    var layer;
    for (layer = 0; layer < this.weights.length; layer += 1) {
      var pre = [];
      var o;
      for (o = 0; o < this.weights[layer].length; o += 1) {
        var sum = this.biases[layer][o];
        var j;
        for (j = 0; j < act.length; j += 1) {
          sum += this.weights[layer][o][j] * act[j];
        }
        pre.push(sum < -50 ? -50 : sum > 50 ? 50 : sum);
      }
      this.preActivations.push(pre);
      if (layer < this.weights.length - 1) {
        act = pre.map(function (v) { return v > 0 ? v : 0; });
      } else {
        act = pre.slice();
      }
      this.activations.push(act);
    }
    return sigmoid(act[0]);
  };

  ValueNeuralNet.prototype.probRedFromBoard = function (board) {
    return this.forward(boardToRedInput(board));
  };

  ValueNeuralNet.prototype.trainOnBoard = function (board, target) {
    var pred = this.forward(boardToRedInput(board));
    var err = pred - target;
    var deltaVal = err * pred * (1 - pred);
    var delta = [deltaVal < -5 ? -5 : deltaVal > 5 ? 5 : deltaVal];
    var layer;
    for (layer = this.weights.length - 1; layer >= 0; layer -= 1) {
      var prev = this.activations[layer];
      var wg = outer(delta, prev);
      var r;
      for (r = 0; r < this.weights[layer].length; r += 1) {
        var c;
        for (c = 0; c < this.weights[layer][r].length; c += 1) {
          this.weights[layer][r][c] -= this.learningRate * wg[r][c];
        }
        this.biases[layer][r] -= this.learningRate * delta[r];
      }
      if (layer === 0) {
        break;
      }
      var reluDeriv = [];
      var i;
      for (i = 0; i < this.preActivations[layer - 1].length; i += 1) {
        reluDeriv.push(this.preActivations[layer - 1][i] > 0 ? 1 : 0);
      }
      var next = matVecT(this.weights[layer], delta);
      delta = [];
      for (i = 0; i < next.length; i += 1) {
        var v = next[i] * reluDeriv[i];
        delta.push(v < -5 ? -5 : v > 5 ? 5 : v);
      }
    }
    clipWeights(this);
  };

  ValueNeuralNet.prototype.toDict = function () {
    return {
      layerSizes: this.layerSizes,
      weights: this.weights,
      biases: this.biases,
      learningRate: this.learningRate
    };
  };

  ValueNeuralNet.fromDict = function (data) {
    var net = new ValueNeuralNet(data && data.layerSizes, data && data.learningRate);
    if (data) {
      net.weights = data.weights || net.weights;
      net.biases = data.biases || net.biases;
      if (typeof data.learningRate === "number") {
        net.learningRate = data.learningRate;
      }
    }
    return net;
  };

  function nnExploreRate(gamesPlayed) {
    var g = typeof gamesPlayed === "number" && gamesPlayed > 0 ? gamesPlayed : 0;
    var t = Math.min(1, g / NN_EXPLORE_DECAY_GAMES);
    return NN_EXPLORE_RATE_START + (NN_EXPLORE_RATE_MIN - NN_EXPLORE_RATE_START) * t;
  }

  function gaFitnessGames() {
    return C.NEAT_FITNESS_GAMES || 10;
  }

  function pickPeerIndex(popSize, selfIndex) {
    if (popSize < 2) {
      return selfIndex;
    }
    var j = Math.floor(Math.random() * (popSize - 1));
    if (j >= selfIndex) {
      j += 1;
    }
    return j;
  }

  function fitnessPoints(result) {
    if (result === "win") {
      return 1;
    }
    if (result === "draw") {
      return 0.5;
    }
    return 0;
  }

  function sampleSoftmaxFromLogits(outputs, moves, temperature) {
    if (!moves.length) {
      return null;
    }
    if (moves.length === 1) {
      return moves[0];
    }
    var temp = temperature > 1e-6 ? temperature : 1e-6;
    var max = -Infinity;
    var i;
    for (i = 0; i < moves.length; i += 1) {
      var v = outputs[moves[i]] / temp;
      if (v > max) {
        max = v;
      }
    }
    var weights = [];
    var sum = 0;
    for (i = 0; i < moves.length; i += 1) {
      var w = Math.exp(outputs[moves[i]] / temp - max);
      weights.push(w);
      sum += w;
    }
    var r = Math.random() * sum;
    var acc = 0;
    for (i = 0; i < moves.length; i += 1) {
      acc += weights[i];
      if (r <= acc) {
        return moves[i];
      }
    }
    return moves[moves.length - 1];
  }

  function pickUniformExploreMove(board, explore, gamesPlayed) {
    if (!explore || Math.random() >= nnExploreRate(gamesPlayed)) {
      return null;
    }
    var moves = G.legalMoves(board);
    if (!moves.length) {
      return null;
    }
    return moves[Math.floor(Math.random() * moves.length)];
  }

  /** When exploring, sample legal cols ~ softmax(logits/T) instead of uniform random. */
  function pickPolicyExploreMove(net, board, player, explore, gamesPlayed) {
    if (!explore || Math.random() >= nnExploreRate(gamesPlayed)) {
      return null;
    }
    var moves = G.legalMoves(board);
    if (!moves.length) {
      return null;
    }
    var outputs = net.forward(boardToInput(board, player));
    return sampleSoftmaxFromLogits(outputs, moves, NN_EXPLORE_TEMPERATURE);
  }

  function pickRandomExploreMove(board, explore, exploreRate) {
    /* Legacy helper kept for any callers; prefer pickUniformExploreMove. */
    if (explore && Math.random() < exploreRate) {
      var moves = G.legalMoves(board);
      return moves[Math.floor(Math.random() * moves.length)];
    }
    return null;
  }

  function maxPolicyLogit(net, board, player) {
    var moves = G.legalMoves(board);
    if (!moves.length) {
      return 0;
    }
    var outputs = net.forward(boardToInput(board, player));
    var best = -Infinity;
    var i;
    for (i = 0; i < moves.length; i += 1) {
      var move = moves[i];
      if (outputs[move] > best) {
        best = outputs[move];
      }
    }
    return best;
  }

  function policyNetPRedWin(net, board) {
    if (net && typeof net.probRedFromBoard === "function") {
      return net.probRedFromBoard(board);
    }
    return sigmoid(maxPolicyLogit(net, board, X));
  }

  function policyGenomePRedWin(genome, board) {
    return sigmoid(maxPolicyLogit(genome.net, board, X));
  }

  function chooseMoveNnTraining(probRed, board, player, explore, gamesPlayed) {
    var randomMove = pickUniformExploreMove(board, explore, gamesPlayed);
    if (randomMove !== null) {
      return randomMove;
    }
    return chooseMoveValueMinimax(probRed, board, player, VALUE_MINIMAX_DEPTH);
  }

  function chooseMovePolicyTraining(net, board, player, explore, gamesPlayed) {
    var randomMove = pickPolicyExploreMove(net, board, player, explore, gamesPlayed);
    if (randomMove !== null) {
      return randomMove;
    }
    return chooseMoveValueMinimax(function (b) {
      return policyNetPRedWin(net, b);
    }, board, player, VALUE_MINIMAX_DEPTH);
  }

  /** One forward + argmax over legal cols (cheap training path for TF.js). */
  function chooseMovePolicyGreedy(net, board, player, explore, gamesPlayed) {
    var randomMove = pickPolicyExploreMove(net, board, player, explore, gamesPlayed);
    if (randomMove !== null) {
      return randomMove;
    }
    var moves = G.legalMoves(board);
    if (!moves.length) {
      return null;
    }
    var outputs = net.forward(boardToInput(board, player));
    return NN.pickMoveFromOutputs(outputs, moves, false);
  }

  function policyScoreFromOutputs(outputs, legalCols) {
    var best = -Infinity;
    var i;
    if (!legalCols.length) {
      return 0.5;
    }
    for (i = 0; i < legalCols.length; i += 1) {
      var v = outputs[legalCols[i]];
      if (v > best) {
        best = v;
      }
    }
    return sigmoid(best);
  }

  /**
   * Same 1-ply search as chooseMovePolicyTraining, but batches child-board
   * net evals into one forward. Prefers a real value head when present.
   */
  function chooseMovePolicyOnePlyBatched(net, board, player, explore, gamesPlayed) {
    var randomMove = pickPolicyExploreMove(net, board, player, explore, gamesPlayed);
    if (randomMove !== null) {
      return randomMove;
    }
    var moves = G.legalMoves(board);
    if (!moves.length) {
      return null;
    }
    if (moves.length === 1) {
      return moves[0];
    }
    if (typeof net.getOutputsBatch !== "function" && typeof net.getValueBatch !== "function") {
      return chooseMovePolicyTraining(net, board, player, false);
    }
    var scratch = G.cloneBoard(board);
    var validCols = [];
    var scores = [];
    var batchInputs = [];
    var batchChildLegal = [];
    var batchIndex = [];
    var i;
    for (i = 0; i < moves.length; i += 1) {
      var col = moves[i];
      var token = G.makeMove(scratch, player, col);
      if (!token) {
        continue;
      }
      var idx = validCols.length;
      validCols.push(col);
      var winner = G.findWinner(scratch);
      if (winner === X) {
        scores.push(1);
      } else if (winner === O) {
        scores.push(0);
      } else if (G.isDraw(scratch)) {
        scores.push(0.5);
      } else {
        scores.push(null);
        batchIndex.push(idx);
        batchInputs.push(boardToInput(scratch, X));
        batchChildLegal.push(G.legalMoves(scratch).slice());
      }
      G.undoMove(scratch, token);
    }
    if (batchInputs.length) {
      if (typeof net.getValueBatch === "function") {
        var vals = net.getValueBatch(batchInputs);
        for (i = 0; i < batchIndex.length; i += 1) {
          scores[batchIndex[i]] = vals[i];
        }
      } else {
        var outs = net.getOutputsBatch(batchInputs);
        for (i = 0; i < batchIndex.length; i += 1) {
          scores[batchIndex[i]] = policyScoreFromOutputs(outs[i], batchChildLegal[i]);
        }
      }
    }
    var maximizing = player === X;
    var bestScore = maximizing ? -Infinity : Infinity;
    var picks = [];
    for (i = 0; i < validCols.length; i += 1) {
      var sc = scores[i];
      if (typeof sc !== "number") {
        continue;
      }
      if (maximizing) {
        if (sc > bestScore) {
          bestScore = sc;
          picks = [validCols[i]];
        } else if (sc === bestScore) {
          picks.push(validCols[i]);
        }
      } else if (sc < bestScore) {
        bestScore = sc;
        picks = [validCols[i]];
      } else if (sc === bestScore) {
        picks.push(validCols[i]);
      }
    }
    if (!picks.length) {
      return moves[0];
    }
    return picks[Math.floor(Math.random() * picks.length)];
  }

  function chooseMoveValueMinimaxTournament(probRed, board, player, batchCtx) {
    return chooseMoveValueMinimax(probRed, board, player, NN_TOURNAMENT_DEPTH, batchCtx);
  }

  function chooseMovePolicyTournament(net, board, player) {
    return chooseMoveValueMinimaxTournament(function (b) {
      return policyNetPRedWin(net, b);
    }, board, player, tfBatchCtxFromNet(net, true));
  }

  /**
   * Depth-1 node: score every child in one net batch (big TF dataSync win).
   * encodeBoard(board) -> input vector; scoreBatch(inputs) -> scores[].
   */
  function alphaBetaDepthOneBatched(
    board, alpha, beta, sideToMove, probRed, encodeBoard, scoreBatch, timeCtx
  ) {
    if (timeCtx && performance.now() >= timeCtx.deadline) {
      timeCtx.aborted = true;
      return { score: probRed(board), move: null };
    }
    var winner = G.findWinner(board);
    if (winner === X) {
      return { score: 1, move: null };
    }
    if (winner === O) {
      return { score: 0, move: null };
    }
    if (G.isDraw(board)) {
      return { score: 0.5, move: null };
    }
    var moves = G.legalMoves(board).slice();
    if (!moves.length) {
      return { score: probRed(board), move: null };
    }
    moves.sort(function () { return Math.random() - 0.5; });
    var validCols = [];
    var scores = [];
    var batchInputs = [];
    var batchIndex = [];
    var i;
    for (i = 0; i < moves.length; i += 1) {
      if (timeCtx && performance.now() >= timeCtx.deadline) {
        timeCtx.aborted = true;
        break;
      }
      var token = G.makeMove(board, sideToMove, moves[i]);
      if (!token) {
        continue;
      }
      var idx = validCols.length;
      validCols.push(moves[i]);
      var w = G.findWinner(board);
      if (w === X) {
        scores.push(1);
      } else if (w === O) {
        scores.push(0);
      } else if (G.isDraw(board)) {
        scores.push(0.5);
      } else {
        scores.push(null);
        batchIndex.push(idx);
        batchInputs.push(encodeBoard(board));
      }
      G.undoMove(board, token);
    }
    if (batchInputs.length) {
      var vals = scoreBatch(batchInputs);
      for (i = 0; i < batchIndex.length; i += 1) {
        scores[batchIndex[i]] = vals[i];
      }
    }
    var maximizing = sideToMove === X;
    var bestScore = maximizing ? -Infinity : Infinity;
    var bestMove = validCols.length ? validCols[0] : moves[0];
    for (i = 0; i < validCols.length; i += 1) {
      var sc = scores[i];
      if (typeof sc !== "number") {
        continue;
      }
      if (maximizing) {
        if (sc > bestScore) {
          bestScore = sc;
          bestMove = validCols[i];
        }
      } else if (sc < bestScore) {
        bestScore = sc;
        bestMove = validCols[i];
      }
    }
    if (bestScore === -Infinity || bestScore === Infinity) {
      return { score: probRed(board), move: bestMove };
    }
    return { score: bestScore, move: bestMove };
  }

  function tfBatchCtxFromNet(net, policyEncodesWithSide) {
    if (!net || typeof net.getValueBatch !== "function") {
      return null;
    }
    return {
      encodeBoard: policyEncodesWithSide ?
        function (b) { return boardToInput(b, X); } :
        function (b) { return boardToRedInput(b); },
      scoreBatch: function (inputs) {
        return net.getValueBatch(inputs);
      }
    };
  }

  function chooseMoveValueMinimaxTimed(probRed, board, player, budgetMs, batchCtx, maxDepth) {
    var moves = G.legalMoves(board);
    if (!moves.length) {
      return { move: null, depth: 0 };
    }
    if (moves.length === 1) {
      return { move: moves[0], depth: 0 };
    }
    var depthCap = typeof maxDepth === "number" && maxDepth > 0 ?
      Math.min(Math.floor(maxDepth), C.CELLS) : C.CELLS;
    var searchBoard = G.cloneBoard(board);
    var hasBudget = typeof budgetMs === "number" && budgetMs > 0 && isFinite(budgetMs);
    var deadline = hasBudget ? performance.now() + budgetMs : Infinity;
    var bestMove = moves[0];
    var depth = 1;
    var completedDepth = 0;
    var timeCtx = hasBudget ? { deadline: deadline, aborted: false } : null;
    while (depth <= depthCap && performance.now() < deadline) {
      if (timeCtx) {
        timeCtx.aborted = false;
      }
      var res = alphaBetaValue(
        searchBoard, depth, -Infinity, Infinity, player, probRed, timeCtx, batchCtx
      );
      if (timeCtx && timeCtx.aborted) {
        break;
      }
      if (res.move !== null) {
        bestMove = res.move;
      }
      completedDepth = depth;
      depth += 1;
    }
    return { move: bestMove, depth: completedDepth };
  }

  function takeTimedMove(agent, timed) {
    if (!timed || typeof timed !== "object") {
      agent.lastDepth = 0;
      return timed;
    }
    agent.lastDepth = timed.depth || 0;
    return timed.move;
  }

  function chooseMovePolicyTimed(net, board, player, budgetMs, maxDepth) {
    return chooseMoveValueMinimaxTimed(function (b) {
      return policyNetPRedWin(net, b);
    }, board, player, budgetMs, tfBatchCtxFromNet(net, true), maxDepth);
  }

  /* NN/value minimax: in-place make/undo only; see makeMove comment in game.js (~40x vs applyMove). */

  function alphaBetaValue(board, depth, alpha, beta, sideToMove, probRed, timeCtx, batchCtx) {
    if (timeCtx && performance.now() >= timeCtx.deadline) {
      timeCtx.aborted = true;
      return { score: probRed(board), move: null };
    }
    var winner = G.findWinner(board);
    if (winner === X) {
      return { score: 1, move: null };
    }
    if (winner === O) {
      return { score: 0, move: null };
    }
    if (G.isDraw(board)) {
      return { score: 0.5, move: null };
    }
    if (depth === 0) {
      return { score: probRed(board), move: null };
    }
    if (
      depth === 1 &&
      batchCtx &&
      typeof batchCtx.encodeBoard === "function" &&
      typeof batchCtx.scoreBatch === "function"
    ) {
      return alphaBetaDepthOneBatched(
        board, alpha, beta, sideToMove, probRed,
        batchCtx.encodeBoard, batchCtx.scoreBatch, timeCtx
      );
    }
    var moves = G.legalMoves(board).slice();
    if (!moves.length) {
      return { score: probRed(board), move: null };
    }
    moves.sort(function () { return Math.random() - 0.5; });
    var maximizing = sideToMove === X;
    var bestMove = moves[0];
    if (maximizing) {
      var value = -Infinity;
      var i;
      var token;
      for (i = 0; i < moves.length; i += 1) {
        if (timeCtx && timeCtx.aborted) {
          break;
        }
        token = G.makeMove(board, sideToMove, moves[i]);
        if (!token) {
          continue;
        }
        var res = alphaBetaValue(
          board, depth - 1, alpha, beta, G.other(sideToMove), probRed, timeCtx, batchCtx
        );
        G.undoMove(board, token);
        if (res.score > value) {
          value = res.score;
          bestMove = moves[i];
        }
        alpha = Math.max(alpha, value);
        if (beta <= alpha) {
          break;
        }
      }
      return { score: value, move: bestMove };
    }
    value = Infinity;
    for (i = 0; i < moves.length; i += 1) {
      if (timeCtx && timeCtx.aborted) {
        break;
      }
      token = G.makeMove(board, sideToMove, moves[i]);
      if (!token) {
        continue;
      }
      res = alphaBetaValue(
        board, depth - 1, alpha, beta, G.other(sideToMove), probRed, timeCtx, batchCtx
      );
      G.undoMove(board, token);
      if (res.score < value) {
        value = res.score;
        bestMove = moves[i];
      }
      beta = Math.min(beta, value);
      if (beta <= alpha) {
        break;
      }
    }
    return { score: value, move: bestMove };
  }

  function chooseMoveValueMinimax(probRed, board, player, depth, batchCtx) {
    var d = typeof depth === "number" ? depth : VALUE_MINIMAX_DEPTH;
    var searchBoard = G.cloneBoard(board);
    var res = alphaBetaValue(searchBoard, d, -Infinity, Infinity, player, probRed, null, batchCtx);
    if (res.move === null) {
      return G.legalMoves(board)[0];
    }
    return res.move;
  }

  /* --- SarsaBrain --- */
  function SarsaBrain() {
    OPP.QBrain.call(this);
  }
  SarsaBrain.prototype = Object.create(OPP.QBrain.prototype);
  SarsaBrain.prototype.constructor = SarsaBrain;

  /* 5-arg (boardKeyFn, legalMovesFn, explore) is the tabular path.
     3-arg (explore as 3rd) is the play / agent.chooseMove API. */
  SarsaBrain.prototype.chooseMove = function (board, player, boardKeyFn, legalMovesFn, explore) {
    if (typeof boardKeyFn === "function") {
      return OPP.QBrain.prototype.chooseMove.call(
        this, board, player, boardKeyFn, legalMovesFn, explore
      );
    }
    return OPP.QBrain.prototype.chooseMove.call(
      this, board, player, G.boardKey, G.legalMoves, boardKeyFn
    );
  };

  /* Same reverse max-Q update as QBrain (old forward SARSA left early plies
     learning almost nothing vs Random). */
  SarsaBrain.prototype.learnFromTrajectory = function (trajectory, mark, winner) {
    return OPP.QBrain.prototype.learnFromTrajectory.call(this, trajectory, mark, winner);
  };

  SarsaBrain.fromDict = function (data) {
    var brain = new SarsaBrain();
    brain.load(data);
    return brain;
  };

  /* --- NnAgent --- */
  function NnAgent(layerSizes, learningRate) {
    this.net = new NN.NeuralNet(layerSizes || NN_POLICY_LAYER_SIZES[2] || NN.NN_LAYER_SIZES, learningRate);
    this.learningRate = this.net.learningRate;
    this.trainSteps = 0;
  }

  NnAgent.prototype.chooseMove = function (board, player, explore, gamesPlayed) {
    return chooseMovePolicyTraining(this.net, board, player, explore, gamesPlayed);
  };

  NnAgent.prototype.chooseMoveTimed = function (board, player, budgetMs, maxDepth) {
    return takeTimedMove(this, chooseMovePolicyTimed(this.net, board, player, budgetMs, maxDepth));
  };

  NnAgent.prototype.chooseMoveMinimax = function (board, player) {
    return chooseMovePolicyTournament(this.net, board, player);
  };

  NnAgent.prototype.learnFromTrajectory = function (trajectory, mark, winner) {
    var reward = 0;
    if (winner === mark) {
      reward = REWARD_WIN;
    } else if (winner && winner !== mark) {
      reward = REWARD_LOSS;
    }
    var steps = G.expandTrajectoryWithMirrors(trajectory);
    var i;
    for (i = 0; i < steps.length; i += 1) {
      this.net.trainOnAction(steps[i].input, steps[i].action, reward);
      this.trainSteps += 1;
    }
  };

  NnAgent.prototype.buildStep = function (board, player, action) {
    return {
      input: boardToInput(board, player),
      action: action,
      board: cloneBoard(board),
      player: player
    };
  };

  NnAgent.prototype.toDict = function () {
    return {
      net: this.net.toJSON(),
      trainSteps: this.trainSteps,
      learningRate: this.learningRate
    };
  };

  NnAgent.fromDict = function (data) {
    var agent = new NnAgent(undefined, data && data.learningRate);
    if (data) {
      agent.net.load(data.net);
      if (typeof data.learningRate === "number") {
        agent.learningRate = data.learningRate;
        agent.net.learningRate = data.learningRate;
      }
      agent.trainSteps = data.trainSteps || 0;
    }
    return agent;
  };

  /* --- TfjsPolicyAgent / TfjsValueAgent (TensorFlow.js or JS fallback) --- */
  var TfjsPolicyAgent = TFJS && TFJS.TfjsPolicyAgent ? TFJS.TfjsPolicyAgent : null;
  var TfjsValueAgent = TFJS && TFJS.TfjsValueAgent ? TFJS.TfjsValueAgent : null;

  if (TfjsPolicyAgent) {
    TfjsPolicyAgent.prototype.chooseMove = function (board, player, explore, gamesPlayed) {
      /* 1-ply policy search with batched child evals (fewer TF dataSyncs). */
      return chooseMovePolicyOnePlyBatched(this.net, board, player, explore, gamesPlayed);
    };

    TfjsPolicyAgent.prototype.chooseMoveTimed = function (board, player, budgetMs, maxDepth) {
      return takeTimedMove(this, chooseMovePolicyTimed(this.net, board, player, budgetMs, maxDepth));
    };

    TfjsPolicyAgent.prototype.chooseMoveMinimax = function (board, player) {
      return chooseMovePolicyTournament(this.net, board, player);
    };
  }

  /**
   * 1-ply search using a value net's getValueBatch (red-view board inputs).
   */
  function chooseMoveValueOnePlyBatched(net, board, player, explore, gamesPlayed) {
    var randomMove = pickUniformExploreMove(board, explore, gamesPlayed);
    if (randomMove !== null) {
      return randomMove;
    }
    var moves = G.legalMoves(board);
    if (!moves.length) {
      return null;
    }
    if (moves.length === 1) {
      return moves[0];
    }
    if (typeof net.getValueBatch !== "function") {
      return chooseMoveNnTraining(function (b) {
        return net.probRedFromBoard(b);
      }, board, player, false, gamesPlayed);
    }
    var scratch = G.cloneBoard(board);
    var validCols = [];
    var scores = [];
    var batchInputs = [];
    var batchIndex = [];
    var i;
    for (i = 0; i < moves.length; i += 1) {
      var col = moves[i];
      var token = G.makeMove(scratch, player, col);
      if (!token) {
        continue;
      }
      var idx = validCols.length;
      validCols.push(col);
      var winner = G.findWinner(scratch);
      if (winner === X) {
        scores.push(1);
      } else if (winner === O) {
        scores.push(0);
      } else if (G.isDraw(scratch)) {
        scores.push(0.5);
      } else {
        scores.push(null);
        batchIndex.push(idx);
        batchInputs.push(boardToRedInput(scratch));
      }
      G.undoMove(scratch, token);
    }
    if (batchInputs.length) {
      var vals = net.getValueBatch(batchInputs);
      for (i = 0; i < batchIndex.length; i += 1) {
        scores[batchIndex[i]] = vals[i];
      }
    }
    var maximizing = player === X;
    var bestScore = maximizing ? -Infinity : Infinity;
    var picks = [];
    for (i = 0; i < validCols.length; i += 1) {
      var sc = scores[i];
      if (typeof sc !== "number") {
        continue;
      }
      if (maximizing) {
        if (sc > bestScore) {
          bestScore = sc;
          picks = [validCols[i]];
        } else if (sc === bestScore) {
          picks.push(validCols[i]);
        }
      } else if (sc < bestScore) {
        bestScore = sc;
        picks = [validCols[i]];
      } else if (sc === bestScore) {
        picks.push(validCols[i]);
      }
    }
    if (!picks.length) {
      return moves[0];
    }
    return picks[Math.floor(Math.random() * picks.length)];
  }

  if (TfjsValueAgent) {
    TfjsValueAgent.prototype.chooseMove = function (board, player, explore, gamesPlayed) {
      return chooseMoveValueOnePlyBatched(this.net, board, player, explore, gamesPlayed);
    };

    TfjsValueAgent.prototype.chooseMoveTimed = function (board, player, budgetMs, maxDepth) {
      var self = this;
      return takeTimedMove(this, chooseMoveValueMinimaxTimed(
        function (b) { return self.probRedWins(b); },
        board,
        player,
        budgetMs,
        tfBatchCtxFromNet(self.net, false),
        maxDepth
      ));
    };

    TfjsValueAgent.prototype.chooseMoveMinimax = function (board, player) {
      var self = this;
      return chooseMoveValueMinimaxTournament(
        function (b) { return self.probRedWins(b); },
        board,
        player,
        tfBatchCtxFromNet(self.net, false)
      );
    };
  }

  /* --- NnValueAgent --- */
  function NnValueAgent(layerSizes, learningRate) {
    this.net = new ValueNeuralNet(layerSizes || NN_VALUE_LAYER_SIZES[2], learningRate);
    this.learningRate = this.net.learningRate;
    this.trainSteps = 0;
  }

  NnValueAgent.prototype.probRedWins = function (board) {
    return this.net.probRedFromBoard(board);
  };

  NnValueAgent.prototype.chooseMove = function (board, player, explore, gamesPlayed) {
    var self = this;
    return chooseMoveNnTraining(function (b) { return self.probRedWins(b); }, board, player, explore, gamesPlayed);
  };

  NnValueAgent.prototype.chooseMoveTimed = function (board, player, budgetMs, maxDepth) {
    var self = this;
    return takeTimedMove(this, chooseMoveValueMinimaxTimed(
      function (b) { return self.probRedWins(b); },
      board,
      player,
      budgetMs,
      null,
      maxDepth
    ));
  };

  NnValueAgent.prototype.chooseMoveMinimax = function (board, player) {
    var self = this;
    return chooseMoveValueMinimaxTournament(function (b) { return self.probRedWins(b); }, board, player);
  };

  NnValueAgent.prototype.learnFromTrajectory = function (trajectory, mark, winner) {
    var target = redWinTarget(winner);
    var steps = G.expandTrajectoryWithMirrors(trajectory);
    var i;
    for (i = 0; i < steps.length; i += 1) {
      this.net.trainOnBoard(steps[i].board, target);
      this.trainSteps += 1;
    }
  };

  NnValueAgent.prototype.buildStep = function (board, player, action) {
    return { board: cloneBoard(board) };
  };

  NnValueAgent.prototype.toDict = function () {
    return {
      net: this.net.toDict(),
      trainSteps: this.trainSteps,
      learningRate: this.learningRate
    };
  };

  NnValueAgent.fromDict = function (data) {
    var agent = new NnValueAgent(undefined, data && data.learningRate);
    if (data) {
      agent.net = ValueNeuralNet.fromDict(data.net);
      if (typeof data.learningRate === "number") {
        agent.learningRate = data.learningRate;
        agent.net.learningRate = data.learningRate;
      }
      agent.trainSteps = data.trainSteps || 0;
    }
    return agent;
  };

  /* --- ReinforceAgent --- */
  function ReinforceAgent(learningRate, layerSizes) {
    this.net = new NN.NeuralNet(layerSizes || NN_POLICY_LAYER_SIZES[2] || NN.NN_LAYER_SIZES);
    this.learningRate = typeof learningRate === "number" ? learningRate : 0.01;
    this.trainSteps = 0;
    this.lastLoss = 0;
  }

  ReinforceAgent.prototype.softmax = function (logits, moves) {
    var subset = [];
    var max = -Infinity;
    var i;
    for (i = 0; i < moves.length; i += 1) {
      var v = logits[moves[i]];
      subset.push(v);
      if (v > max) {
        max = v;
      }
    }
    var exp = subset.map(function (v) { return Math.exp(v - max); });
    var sum = 0;
    for (i = 0; i < exp.length; i += 1) {
      sum += exp[i];
    }
    var out = [];
    for (i = 0; i < logits.length; i += 1) {
      out.push(0);
    }
    for (i = 0; i < moves.length; i += 1) {
      out[moves[i]] = exp[i] / sum;
    }
    return out;
  };

  ReinforceAgent.prototype.chooseMove = function (board, player, explore, gamesPlayed) {
    return chooseMovePolicyTraining(this.net, board, player, explore, gamesPlayed);
  };

  ReinforceAgent.prototype.chooseMoveTimed = function (board, player, budgetMs, maxDepth) {
    return takeTimedMove(this, chooseMovePolicyTimed(this.net, board, player, budgetMs, maxDepth));
  };

  ReinforceAgent.prototype.chooseMoveMinimax = function (board, player) {
    return chooseMovePolicyTournament(this.net, board, player);
  };

  ReinforceAgent.prototype.applyPolicyGrad = function (gradOutput, reward) {
    var delta = gradOutput.map(function (g) { return g * reward * this.learningRate; }.bind(this));
    var layer;
    for (layer = this.net.weights.length - 1; layer >= 0; layer -= 1) {
      var prev = this.net.activations[layer];
      var r;
      for (r = 0; r < delta.length; r += 1) {
        var c;
        for (c = 0; c < prev.length; c += 1) {
          this.net.weights[layer][r][c] -= delta[r] * prev[c];
        }
        this.net.biases[layer][r] -= delta[r];
      }
      if (layer === 0) {
        break;
      }
      var reluDeriv = [];
      for (c = 0; c < this.net.preActivations[layer - 1].length; c += 1) {
        reluDeriv.push(this.net.preActivations[layer - 1][c] > 0 ? 1 : 0);
      }
      var next = matVecT(this.net.weights[layer], delta);
      delta = [];
      for (c = 0; c < next.length; c += 1) {
        delta.push(next[c] * reluDeriv[c]);
      }
    }
  };

  ReinforceAgent.prototype.learnFromTrajectory = function (trajectory, mark, winner) {
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
    var lossSum = 0;
    var i;
    for (i = 0; i < steps.length; i += 1) {
      var step = steps[i];
      var logits = this.net.forward(step.input);
      var probs = this.softmax(logits, step.moves);
      var grad = probs.slice();
      grad[step.action] -= 1;
      lossSum += -Math.log(probs[step.action] < 1e-8 ? 1e-8 : probs[step.action]) * reward;
      this.applyPolicyGrad(grad, reward);
      this.trainSteps += 1;
    }
    this.lastLoss = lossSum / steps.length;
  };

  ReinforceAgent.prototype.buildStep = function (board, player, action) {
    return {
      input: boardToInput(board, player),
      action: action,
      moves: G.legalMoves(board),
      board: cloneBoard(board),
      player: player
    };
  };

  ReinforceAgent.prototype.toDict = function () {
    return {
      net: this.net.toJSON(),
      learningRate: this.learningRate,
      trainSteps: this.trainSteps,
      lastLoss: this.lastLoss
    };
  };

  ReinforceAgent.fromDict = function (data) {
    var agent = new ReinforceAgent();
    if (data) {
      agent.net.load(data.net);
      agent.learningRate = typeof data.learningRate === "number" ? data.learningRate : 0.01;
      agent.trainSteps = data.trainSteps || 0;
      agent.lastLoss = data.lastLoss || 0;
    }
    return agent;
  };

  /* --- ReinforceValueAgent --- */
  function ReinforceValueAgent(layerSizes, learningRate) {
    this.net = new ValueNeuralNet(layerSizes || NN_VALUE_LAYER_SIZES[2], learningRate);
    this.learningRate = this.net.learningRate;
    this.trainSteps = 0;
  }

  ReinforceValueAgent.prototype.probRedWins = function (board) {
    return this.net.probRedFromBoard(board);
  };

  ReinforceValueAgent.prototype.chooseMove = function (board, player, explore, gamesPlayed) {
    var self = this;
    return chooseMoveNnTraining(function (b) { return self.probRedWins(b); }, board, player, explore, gamesPlayed);
  };

  ReinforceValueAgent.prototype.chooseMoveTimed = function (board, player, budgetMs, maxDepth) {
    var self = this;
    return takeTimedMove(this, chooseMoveValueMinimaxTimed(
      function (b) { return self.probRedWins(b); },
      board,
      player,
      budgetMs,
      null,
      maxDepth
    ));
  };

  ReinforceValueAgent.prototype.chooseMoveMinimax = function (board, player) {
    var self = this;
    return chooseMoveValueMinimaxTournament(function (b) { return self.probRedWins(b); }, board, player);
  };

  ReinforceValueAgent.prototype.learnFromTrajectory = function (trajectory, mark, winner) {
    var target = redWinTarget(winner);
    var steps = G.expandTrajectoryWithMirrors(trajectory);
    var i;
    for (i = 0; i < steps.length; i += 1) {
      this.net.trainOnBoard(steps[i].board, target);
      this.trainSteps += 1;
    }
  };

  ReinforceValueAgent.prototype.buildStep = function (board, player, action) {
    return { board: cloneBoard(board) };
  };

  ReinforceValueAgent.prototype.toDict = function () {
    return {
      net: this.net.toDict(),
      trainSteps: this.trainSteps,
      learningRate: this.learningRate
    };
  };

  ReinforceValueAgent.fromDict = function (data) {
    var agent = new ReinforceValueAgent(undefined, data && data.learningRate);
    if (data) {
      agent.net = ValueNeuralNet.fromDict(data.net);
      if (typeof data.learningRate === "number") {
        agent.learningRate = data.learningRate;
        agent.net.learningRate = data.learningRate;
      }
      agent.trainSteps = data.trainSteps || 0;
    }
    return agent;
  };

  /* --- SimpleGaNeatGenome helpers --- */
  function createGaGenome(layerSizes) {
    return {
      layerSizes: layerSizes.slice(),
      net: new NN.NeuralNet(layerSizes)
    };
  }

  function genomeActivate(genome, input) {
    return genome.net.forward(input);
  }

  function genomeCopy(genome) {
    var copy = createGaGenome(genome.layerSizes);
    copy.net.load(genome.net.toJSON());
    return copy;
  }

  function genomeMutate(genome) {
    var l;
    for (l = 0; l < genome.net.weights.length; l += 1) {
      var r;
      for (r = 0; r < genome.net.weights[l].length; r += 1) {
        var c;
        for (c = 0; c < genome.net.weights[l][r].length; c += 1) {
          if (Math.random() < 0.1) {
            genome.net.weights[l][r][c] += Math.random() * 0.4 - 0.2;
          }
        }
        if (Math.random() < 0.1) {
          genome.net.biases[l][r] += Math.random() * 0.4 - 0.2;
        }
      }
    }
  }

  function genomeCrossover(a, b) {
    var child = genomeCopy(a);
    var l;
    for (l = 0; l < child.net.weights.length; l += 1) {
      var r;
      for (r = 0; r < child.net.weights[l].length; r += 1) {
        var c;
        for (c = 0; c < child.net.weights[l][r].length; c += 1) {
          child.net.weights[l][r][c] = Math.random() < 0.5 ?
            a.net.weights[l][r][c] : b.net.weights[l][r][c];
        }
        child.net.biases[l][r] = Math.random() < 0.5 ?
          a.net.biases[l][r] : b.net.biases[l][r];
      }
    }
    return child;
  }

  function createValueGaGenome(layerSizes) {
    return {
      layerSizes: layerSizes.slice(),
      net: new ValueNeuralNet(layerSizes)
    };
  }

  function valueGenomeCopy(genome) {
    var copy = createValueGaGenome(genome.layerSizes);
    copy.net = ValueNeuralNet.fromDict(genome.net.toDict());
    return copy;
  }

  function valueGenomeMutate(genome) {
    var l;
    for (l = 0; l < genome.net.weights.length; l += 1) {
      var r;
      for (r = 0; r < genome.net.weights[l].length; r += 1) {
        var c;
        for (c = 0; c < genome.net.weights[l][r].length; c += 1) {
          if (Math.random() < 0.1) {
            genome.net.weights[l][r][c] += Math.random() * 0.4 - 0.2;
          }
        }
        if (Math.random() < 0.1) {
          genome.net.biases[l][r] += Math.random() * 0.4 - 0.2;
        }
      }
    }
  }

  function valueGenomeCrossover(a, b) {
    var child = valueGenomeCopy(a);
    var l;
    for (l = 0; l < child.net.weights.length; l += 1) {
      var r;
      for (r = 0; r < child.net.weights[l].length; r += 1) {
        var c;
        for (c = 0; c < child.net.weights[l][r].length; c += 1) {
          child.net.weights[l][r][c] = Math.random() < 0.5 ?
            a.net.weights[l][r][c] : b.net.weights[l][r][c];
        }
        child.net.biases[l][r] = Math.random() < 0.5 ?
          a.net.biases[l][r] : b.net.biases[l][r];
      }
    }
    return child;
  }

  /* --- SimpleGaNeatAgent --- */
  function SimpleGaNeatAgent() {
    this.population = [];
    this.generation = 0;
    this.bestFitness = -Infinity;
    this.bestGenome = null;
    this.bestGenomeB64 = null;
    var i;
    for (i = 0; i < C.NEAT_POP_SIZE; i += 1) {
      this.population.push(createGaGenome(NEAT_GA_LAYER_SIZES));
    }
    this.bestGenome = genomeCopy(this.population[0]);
  }

  SimpleGaNeatAgent.prototype.bestGenomeForPlay = function () {
    return this.bestGenome || this.population[0];
  };

  SimpleGaNeatAgent.prototype.chooseMoveWithGenome = function (genome, board, player, explore, gamesPlayed) {
    var randomMove = pickPolicyExploreMove(genome.net, board, player, explore, gamesPlayed);
    if (randomMove !== null) {
      return randomMove;
    }
    return chooseMoveValueMinimax(function (b) {
      return policyGenomePRedWin(genome, b);
    }, board, player, VALUE_MINIMAX_DEPTH);
  };

  SimpleGaNeatAgent.prototype.chooseMoveWithGenomeTimed = function (genome, board, player, budgetMs, maxDepth) {
    return takeTimedMove(this, chooseMoveValueMinimaxTimed(function (b) {
      return policyGenomePRedWin(genome, b);
    }, board, player, budgetMs, null, maxDepth));
  };

  SimpleGaNeatAgent.prototype.chooseMove = function (board, player, explore, gamesPlayed) {
    return this.chooseMoveWithGenome(this.bestGenomeForPlay(), board, player, explore, gamesPlayed);
  };

  SimpleGaNeatAgent.prototype.chooseMoveTimed = function (board, player, budgetMs, maxDepth) {
    return this.chooseMoveWithGenomeTimed(this.bestGenomeForPlay(), board, player, budgetMs, maxDepth);
  };

  SimpleGaNeatAgent.prototype.chooseMoveMinimax = function (board, player) {
    var genome = this.bestGenomeForPlay();
    return chooseMoveValueMinimaxTournament(function (b) {
      return policyGenomePRedWin(genome, b);
    }, board, player);
  };

  SimpleGaNeatAgent.prototype.evolveOneGeneration = function (fitnessFn) {
    var scores = [];
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      scores.push(fitnessFn(this.population[i], i));
    }
    var bestIdx = 0;
    var bestScore = scores[0];
    for (i = 1; i < scores.length; i += 1) {
      if (scores[i] > bestScore) {
        bestScore = scores[i];
        bestIdx = i;
      }
    }
    if (bestScore > this.bestFitness) {
      this.bestFitness = bestScore;
      this.bestGenome = genomeCopy(this.population[bestIdx]);
      this.bestGenomeB64 = null;
    }
    var ranked = scores.map(function (s, idx) { return { s: s, i: idx }; });
    ranked.sort(function (a, b) { return b.s - a.s; });
    var elite = Math.max(2, Math.floor(C.NEAT_POP_SIZE / 10));
    var next = [];
    for (i = 0; i < elite; i += 1) {
      next.push(genomeCopy(this.population[ranked[i].i]));
    }
    while (next.length < C.NEAT_POP_SIZE) {
      var parentA = this.population[ranked[Math.floor(Math.random() * Math.floor(C.NEAT_POP_SIZE / 2))].i];
      var parentB = this.population[ranked[Math.floor(Math.random() * Math.floor(C.NEAT_POP_SIZE / 2))].i];
      var child = genomeCrossover(parentA, parentB);
      genomeMutate(child);
      next.push(child);
    }
    this.population = next;
    this.generation += 1;
  };

  SimpleGaNeatAgent.prototype.extraStats = function () {
    return "gen=" + this.generation + " best=" +
      (this.bestFitness === -Infinity ? "n/a" : this.bestFitness.toFixed(2));
  };

  SimpleGaNeatAgent.prototype.repopulateFromBest = function () {
    var seed = this.bestGenome || createGaGenome(NEAT_GA_LAYER_SIZES);
    this.population = [];
    var i;
    for (i = 0; i < C.NEAT_POP_SIZE; i += 1) {
      if (i === 0) {
        this.population.push(genomeCopy(seed));
      } else {
        var g = genomeCopy(seed);
        genomeMutate(g);
        this.population.push(g);
      }
    }
  };

  SimpleGaNeatAgent.prototype.toDict = function (compact) {
    if (compact) {
      return {
        generation: this.generation,
        bestFitness: this.bestFitness,
        bestGenomeB64: this.bestGenomeB64,
        bestGenome: this.bestGenome ? this.bestGenome.net.toJSON() : null,
        populationCompact: true
      };
    }
    return {
      generation: this.generation,
      bestFitness: this.bestFitness,
      bestGenomeB64: this.bestGenomeB64,
      population: this.population.map(function (g) { return g.net.toJSON(); }),
      bestGenome: this.bestGenome ? this.bestGenome.net.toJSON() : null
    };
  };

  SimpleGaNeatAgent.fromDict = function (data) {
    var agent = new SimpleGaNeatAgent();
    if (!data) {
      return agent;
    }
    agent.generation = data.generation || 0;
    agent.bestFitness = typeof data.bestFitness === "number" ? data.bestFitness : -Infinity;
    agent.bestGenomeB64 = data.bestGenomeB64 || null;
    if (data.bestGenome) {
      agent.bestGenome = createGaGenome(NEAT_GA_LAYER_SIZES);
      agent.bestGenome.net.load(data.bestGenome);
    }
    if (data.populationCompact || !data.population || !data.population.length) {
      agent.repopulateFromBest();
    } else {
      agent.population = data.population.map(function (netData) {
        var g = createGaGenome(NEAT_GA_LAYER_SIZES);
        g.net.load(netData);
        return g;
      });
      if (!agent.bestGenome && agent.population.length) {
        agent.bestGenome = genomeCopy(agent.population[0]);
      }
    }
    return agent;
  };

  /* --- SimpleGaNeatValueAgent --- */
  function SimpleGaNeatValueAgent() {
    this.population = [];
    this.generation = 0;
    this.bestFitness = -Infinity;
    this.bestGenome = null;
    this.bestGenomeB64 = null;
    var i;
    for (i = 0; i < C.NEAT_POP_SIZE; i += 1) {
      this.population.push(createValueGaGenome(NEAT_VALUE_GA_LAYER_SIZES));
    }
    this.bestGenome = valueGenomeCopy(this.population[0]);
  }

  SimpleGaNeatValueAgent.prototype.bestGenomeForPlay = function () {
    return this.bestGenome || this.population[0];
  };

  SimpleGaNeatValueAgent.prototype.probRedWithGenome = function (genome, board) {
    return genome.net.probRedFromBoard(board);
  };

  SimpleGaNeatValueAgent.prototype.probRedWins = function (board) {
    return this.probRedWithGenome(this.bestGenomeForPlay(), board);
  };

  SimpleGaNeatValueAgent.prototype.chooseMoveWithGenome = function (genome, board, player, explore, gamesPlayed) {
    var self = this;
    return chooseMoveNnTraining(function (b) { return self.probRedWithGenome(genome, b); }, board, player, explore, gamesPlayed);
  };

  SimpleGaNeatValueAgent.prototype.chooseMoveWithGenomeTimed = function (genome, board, player, budgetMs, maxDepth) {
    var self = this;
    return takeTimedMove(this, chooseMoveValueMinimaxTimed(
      function (b) { return self.probRedWithGenome(genome, b); },
      board,
      player,
      budgetMs,
      null,
      maxDepth
    ));
  };

  SimpleGaNeatValueAgent.prototype.chooseMove = function (board, player, explore, gamesPlayed) {
    return this.chooseMoveWithGenome(this.bestGenomeForPlay(), board, player, explore, gamesPlayed);
  };

  SimpleGaNeatValueAgent.prototype.chooseMoveTimed = function (board, player, budgetMs, maxDepth) {
    return this.chooseMoveWithGenomeTimed(this.bestGenomeForPlay(), board, player, budgetMs, maxDepth);
  };

  SimpleGaNeatValueAgent.prototype.chooseMoveMinimax = function (board, player) {
    var self = this;
    var genome = this.bestGenomeForPlay();
    return chooseMoveValueMinimaxTournament(function (b) {
      return self.probRedWithGenome(genome, b);
    }, board, player);
  };

  SimpleGaNeatValueAgent.prototype.evolveOneGeneration = function (fitnessFn) {
    var scores = [];
    var i;
    for (i = 0; i < this.population.length; i += 1) {
      scores.push(fitnessFn(this.population[i], i));
    }
    var bestIdx = 0;
    var bestScore = scores[0];
    for (i = 1; i < scores.length; i += 1) {
      if (scores[i] > bestScore) {
        bestScore = scores[i];
        bestIdx = i;
      }
    }
    if (bestScore > this.bestFitness) {
      this.bestFitness = bestScore;
      this.bestGenome = valueGenomeCopy(this.population[bestIdx]);
      this.bestGenomeB64 = null;
    }
    var ranked = scores.map(function (s, idx) { return { s: s, i: idx }; });
    ranked.sort(function (a, b) { return b.s - a.s; });
    var elite = Math.max(2, Math.floor(C.NEAT_POP_SIZE / 10));
    var next = [];
    for (i = 0; i < elite; i += 1) {
      next.push(valueGenomeCopy(this.population[ranked[i].i]));
    }
    while (next.length < C.NEAT_POP_SIZE) {
      var parentA = this.population[ranked[Math.floor(Math.random() * Math.floor(C.NEAT_POP_SIZE / 2))].i];
      var parentB = this.population[ranked[Math.floor(Math.random() * Math.floor(C.NEAT_POP_SIZE / 2))].i];
      var child = valueGenomeCrossover(parentA, parentB);
      valueGenomeMutate(child);
      next.push(child);
    }
    this.population = next;
    this.generation += 1;
  };

  SimpleGaNeatValueAgent.prototype.extraStats = function () {
    return "gen=" + this.generation + " best=" +
      (this.bestFitness === -Infinity ? "n/a" : this.bestFitness.toFixed(2));
  };

  SimpleGaNeatValueAgent.prototype.repopulateFromBest = function () {
    var seed = this.bestGenome || createValueGaGenome(NEAT_VALUE_GA_LAYER_SIZES);
    this.population = [];
    var i;
    for (i = 0; i < C.NEAT_POP_SIZE; i += 1) {
      if (i === 0) {
        this.population.push(valueGenomeCopy(seed));
      } else {
        var g = valueGenomeCopy(seed);
        valueGenomeMutate(g);
        this.population.push(g);
      }
    }
  };

  SimpleGaNeatValueAgent.prototype.toDict = function (compact) {
    if (compact) {
      return {
        generation: this.generation,
        bestFitness: this.bestFitness,
        bestGenomeB64: this.bestGenomeB64,
        bestGenome: this.bestGenome ? this.bestGenome.net.toDict() : null,
        populationCompact: true
      };
    }
    return {
      generation: this.generation,
      bestFitness: this.bestFitness,
      bestGenomeB64: this.bestGenomeB64,
      population: this.population.map(function (g) { return g.net.toDict(); }),
      bestGenome: this.bestGenome ? this.bestGenome.net.toDict() : null
    };
  };

  SimpleGaNeatValueAgent.fromDict = function (data) {
    var agent = new SimpleGaNeatValueAgent();
    if (!data) {
      return agent;
    }
    agent.generation = data.generation || 0;
    agent.bestFitness = typeof data.bestFitness === "number" ? data.bestFitness : -Infinity;
    agent.bestGenomeB64 = data.bestGenomeB64 || null;
    if (data.bestGenome) {
      agent.bestGenome = createValueGaGenome(NEAT_VALUE_GA_LAYER_SIZES);
      agent.bestGenome.net = ValueNeuralNet.fromDict(data.bestGenome);
    }
    if (data.populationCompact || !data.population || !data.population.length) {
      agent.repopulateFromBest();
    } else {
      agent.population = data.population.map(function (netData) {
        var g = createValueGaGenome(NEAT_VALUE_GA_LAYER_SIZES);
        g.net = ValueNeuralNet.fromDict(netData);
        return g;
      });
      if (!agent.bestGenome && agent.population.length) {
        agent.bestGenome = valueGenomeCopy(agent.population[0]);
      }
    }
    return agent;
  };

  function menaceCopy(brain) {
    var clone = new OPP.MenaceBrain();
    clone.boxes = JSON.parse(JSON.stringify(brain.boxes));
    clone.lastUsedEpisode = {};
    var k;
    for (k in brain.lastUsedEpisode) {
      if (Object.prototype.hasOwnProperty.call(brain.lastUsedEpisode, k)) {
        clone.lastUsedEpisode[k] = brain.lastUsedEpisode[k];
      }
    }
    clone.episodeCounter = brain.episodeCounter;
    return clone;
  }

  /* --- GeneticMenaceAgent --- */
  function GeneticMenaceAgent() {
    this.populationSize = C.GENETIC_MENACE_POP;
    this.population = [];
    this.generation = 0;
    this.bestIndex = 0;
    this.lastBestFitness = 0;
    this.bestFitness = 0;
    var i;
    for (i = 0; i < this.populationSize; i += 1) {
      this.population.push(new OPP.MenaceBrain());
    }
  }

  GeneticMenaceAgent.prototype.champion = function () {
    return this.population[this.bestIndex];
  };

  GeneticMenaceAgent.prototype.chooseMove = function (board, player, explore) {
    return this.champion().chooseMove(board, player, G.boardKey, G.legalMoves, explore);
  };

  GeneticMenaceAgent.prototype.learnFromTrajectory = function (trajectory, winner) {
    this.champion().learnFromTrajectory(trajectory, winner);
  };

  GeneticMenaceAgent.prototype.evolve = function (fitnessScores) {
    if (!fitnessScores.length) {
      return;
    }
    var bestIdx = 0;
    var i;
    for (i = 1; i < fitnessScores.length; i += 1) {
      if (fitnessScores[i] > fitnessScores[bestIdx]) {
        bestIdx = i;
      }
    }
    this.bestIndex = bestIdx;
    this.lastBestFitness = fitnessScores[bestIdx];
    if (fitnessScores[bestIdx] > this.bestFitness) {
      this.bestFitness = fitnessScores[bestIdx];
    }
    var ranked = fitnessScores.map(function (s, idx) { return { s: s, i: idx }; });
    ranked.sort(function (a, b) { return b.s - a.s; });
    var eliteCount = Math.max(2, Math.floor(this.populationSize / 10));
    var next = [];
    for (i = 0; i < eliteCount; i += 1) {
      next.push(menaceCopy(this.population[ranked[i].i]));
    }
    while (next.length < this.populationSize) {
      var parentA = this.population[ranked[Math.floor(Math.random() * Math.floor(this.populationSize / 2))].i];
      var parentB = this.population[ranked[Math.floor(Math.random() * Math.floor(this.populationSize / 2))].i];
      next.push(this.crossover(parentA, parentB));
    }
    this.population = next;
    this.generation += 1;
  };

  GeneticMenaceAgent.prototype.crossover = function (a, b) {
    var child = new OPP.MenaceBrain();
    var keys = {};
    var k;
    for (k in a.boxes) {
      if (Object.prototype.hasOwnProperty.call(a.boxes, k)) {
        keys[k] = true;
      }
    }
    for (k in b.boxes) {
      if (Object.prototype.hasOwnProperty.call(b.boxes, k)) {
        keys[k] = true;
      }
    }
    for (k in keys) {
      if (Object.prototype.hasOwnProperty.call(keys, k)) {
        if (a.boxes[k] && b.boxes[k]) {
          child.boxes[k] = a.boxes[k].map(function (v, idx) {
            return (v + b.boxes[k][idx]) / 2;
          });
        } else if (a.boxes[k]) {
          child.boxes[k] = a.boxes[k].slice();
        } else {
          child.boxes[k] = b.boxes[k].slice();
        }
      }
    }
    this.mutate(child);
    return child;
  };

  GeneticMenaceAgent.prototype.mutate = function (brain) {
    var k;
    for (k in brain.boxes) {
      if (!Object.prototype.hasOwnProperty.call(brain.boxes, k)) {
        continue;
      }
      var beads = brain.boxes[k];
      var i;
      for (i = 0; i < beads.length; i += 1) {
        if (Math.random() < 0.05) {
          beads[i] = Math.max(0, beads[i] + (Math.random() * 0.5 - 0.25));
        }
      }
    }
  };

  GeneticMenaceAgent.prototype.extraStats = function () {
    return "gen=" + this.generation +
      " fit=" + this.lastBestFitness.toFixed(2) +
      " best=" + this.bestFitness.toFixed(2) +
      " boxes=" + this.champion().boxCount();
  };

  GeneticMenaceAgent.prototype.repopulateFromChampion = function () {
    var seed = this.champion();
    this.population = [menaceCopy(seed)];
    while (this.population.length < this.populationSize) {
      var child = menaceCopy(seed);
      this.mutate(child);
      this.population.push(child);
    }
    this.bestIndex = 0;
  };

  GeneticMenaceAgent.prototype.toDict = function (compact) {
    if (compact) {
      return {
        generation: this.generation,
        bestIndex: 0,
        lastBestFitness: this.lastBestFitness,
        bestFitness: this.bestFitness,
        populationSize: this.populationSize,
        championOnly: this.champion().toJSON()
      };
    }
    return {
      population: this.population.map(function (p) { return p.toJSON(); }),
      generation: this.generation,
      bestIndex: this.bestIndex,
      lastBestFitness: this.lastBestFitness,
      bestFitness: this.bestFitness,
      populationSize: this.populationSize
    };
  };

  GeneticMenaceAgent.fromDict = function (data) {
    var agent = new GeneticMenaceAgent();
    if (!data) {
      return agent;
    }
    agent.populationSize = data.populationSize || C.GENETIC_MENACE_POP;
    agent.generation = data.generation || 0;
    agent.bestIndex = data.bestIndex || 0;
    agent.lastBestFitness = data.lastBestFitness || 0;
    agent.bestFitness = typeof data.bestFitness === "number" ? data.bestFitness : (data.lastBestFitness || 0);
    if (data.championOnly) {
      var champ = new OPP.MenaceBrain();
      champ.load(data.championOnly);
      agent.population = [champ];
      agent.repopulateFromChampion();
    } else if (data.population && data.population.length) {
      agent.population = data.population.map(function (p) {
        var b = new OPP.MenaceBrain();
        b.load(p);
        return b;
      });
    }
    return agent;
  };

  /* --- MctsAgent --- */
  function MctsAgent(sims) {
    this.sims = typeof sims === "number" ? sims : C.MCTS_SIMS;
    this.lastSimCount = 0;
  }

  MctsAgent.prototype.chooseMove = function (board, player, explore, budgetMs) {
    var moves = G.legalMoves(board);
    if (moves.length === 1) {
      return moves[0];
    }
    var stats = {};
    var i;
    for (i = 0; i < moves.length; i += 1) {
      stats[moves[i]] = { wins: 0, visits: 0 };
    }
    var totalVisits = 0;
    var simCount = 0;
    var deadline = typeof budgetMs === "number" && budgetMs > 0 ?
      performance.now() + budgetMs : null;
    while (true) {
      if (deadline) {
        if (performance.now() >= deadline) {
          break;
        }
      } else if (simCount >= this.sims) {
        break;
      }
      var move = this.selectMove(stats, moves, totalVisits);
      var wins = this.simulate(board, player, move);
      stats[move].wins += wins;
      stats[move].visits += 1;
      totalVisits += 1;
      simCount += 1;
      if (simCount > 500000) {
        break;
      }
    }
    this.lastSimCount = simCount;
    this.lastBudgetMs = deadline ? budgetMs : null;
    var bestMove = moves[0];
    var bestRate = -Infinity;
    for (i = 0; i < moves.length; i += 1) {
      var m = moves[i];
      var rate = stats[m].wins / (stats[m].visits || 1);
      if (rate > bestRate) {
        bestRate = rate;
        bestMove = m;
      }
    }
    return bestMove;
  };

  MctsAgent.prototype.selectMove = function (stats, moves, totalVisits) {
    if (totalVisits === 0) {
      return moves[Math.floor(Math.random() * moves.length)];
    }
    var logTotal = Math.log(totalVisits < 1 ? 1 : totalVisits);
    var bestScore = -Infinity;
    var picks = [];
    var i;
    for (i = 0; i < moves.length; i += 1) {
      var m = moves[i];
      var visits = stats[m].visits;
      if (visits === 0) {
        return m;
      }
      var winRate = stats[m].wins / visits;
      var ucb = winRate + Math.sqrt(2 * logTotal / visits);
      if (ucb > bestScore) {
        bestScore = ucb;
        picks = [m];
      } else if (ucb === bestScore) {
        picks.push(m);
      }
    }
    return picks[Math.floor(Math.random() * picks.length)];
  };

  MctsAgent.prototype.simulate = function (board, player, firstMove) {
    var b = G.applyMove(board, player, firstMove);
    var current = G.other(player);
    if (G.findWinner(b) === player) {
      return 1;
    }
    if (G.isDraw(b)) {
      return 0.5;
    }
    while (true) {
      var mv = G.randomMove(b);
      b = G.applyMove(b, current, mv);
      var winner = G.findWinner(b);
      if (winner === player) {
        return 1;
      }
      if (winner) {
        return 0;
      }
      if (G.isDraw(b)) {
        return 0.5;
      }
      current = G.other(current);
    }
  };

  MctsAgent.prototype.extraStats = function () {
    if (typeof this.lastBudgetMs === "number") {
      return "budget=" + this.lastBudgetMs.toFixed(2) + "ms last=" + (this.lastSimCount || 0) + " sims";
    }
    return "sims=" + this.sims + " last=" + (this.lastSimCount || 0);
  };

  MctsAgent.prototype.toDict = function () {
    return { sims: this.sims };
  };

  MctsAgent.fromDict = function (data) {
    return new MctsAgent(data && data.sims);
  };

  /* --- MinimaxTournamentAgent --- */
  function MinimaxTournamentAgent() {
    this.lastDepth = 0;
    this.lastBudgetMs = null;
  }

  MinimaxTournamentAgent.prototype.chooseMove = function (board, player, budgetMs) {
    var moves = G.legalMoves(board);
    if (!moves.length) {
      return null;
    }
    if (moves.length === 1) {
      this.lastDepth = 0;
      return moves[0];
    }
    var rng = Math.random.bind(Math);
    var fallbackDepth = C.TOURNAMENT_MINIMAX_MAX_DEPTH || 7;
    if (typeof budgetMs !== "number" || budgetMs <= 0) {
      var fallback = M.alphabeta(board, fallbackDepth, -Infinity, Infinity, player, player, rng);
      this.lastDepth = fallbackDepth;
      this.lastBudgetMs = null;
      return fallback.move !== null ? fallback.move : moves[0];
    }
    var deadline = performance.now() + budgetMs;
    var timed = M.chooseMoveUntilDeadline(board, player, deadline, rng);
    this.lastDepth = timed.depth;
    this.lastBudgetMs = budgetMs;
    return timed.move !== null ? timed.move : moves[0];
  };

  MinimaxTournamentAgent.prototype.extraStats = function () {
    if (typeof this.lastBudgetMs === "number") {
      return "budget=" + this.lastBudgetMs.toFixed(2) + "ms lastDepth=" + (this.lastDepth || 0);
    }
    return "lastDepth=" + (this.lastDepth || 0);
  };

  MinimaxTournamentAgent.prototype.toDict = function () {
    return {};
  };

  MinimaxTournamentAgent.fromDict = function () {
    return new MinimaxTournamentAgent();
  };

  /* --- AlgoState --- */
  function AlgoState(algoId) {
    this.algoId = algoId;
    this.phase = "random";
    this.minimaxDepth = 0;
    this.phaseGamesTotal = 0;
    this.randomGamesTotal = 0;
    this.gamesPlayed = 0;
    this.greedyRecent = [];
    this.nemesis = new OPP.NemesisTracker();
    this.sessionPoints = 0;
    this.droppedAfterTournament = null;
  }

  AlgoState.prototype.isActive = function () {
    return this.droppedAfterTournament === null;
  };

  AlgoState.prototype.greedyWinRate = function () {
    if (!this.greedyRecent.length) {
      return 0;
    }
    var wins = 0;
    var i;
    for (i = 0; i < this.greedyRecent.length; i += 1) {
      if (this.greedyRecent[i] === "win") {
        wins += 1;
      }
    }
    return wins / this.greedyRecent.length;
  };

  AlgoState.prototype.phaseLabel = function () {
    if (this.phase === "random") {
      return "random";
    }
    return "minimax-" + M.normalizeLevel(this.minimaxDepth).toFixed(1);
  };

  AlgoState.prototype.maybeGraduate = function () {
    if (this.greedyRecent.length < C.GREEDY_EVAL_GAMES) {
      return;
    }
    var rate = this.greedyWinRate();
    if (rate > C.GRADUATION_TARGET_WIN_RATE) {
      this.promoteLevel(rate);
    } else if (rate < C.GRADUATION_TARGET_WIN_RATE) {
      this.demoteLevel();
    }
  };

  AlgoState.prototype.promoteLevel = function (greedyRate) {
    if (this.phase === "random") {
      this.phase = "minimax";
      this.minimaxDepth = C.MINIMAX_START_LEVEL;
    } else if (M.normalizeLevel(this.minimaxDepth) >= C.MAX_MINIMAX_LEVEL) {
      return;
    } else {
      var step = C.MINIMAX_LEVEL_STEP;
      if (typeof greedyRate === "number" &&
          greedyRate >= (C.GREEDY_SKIP_LEVEL_WIN_RATE || 0.95)) {
        step = C.MINIMAX_LEVEL_STEP * 2;
      }
      this.minimaxDepth = M.normalizeLevel(this.minimaxDepth + step);
    }
    this.phaseGamesTotal = 0;
  };

  AlgoState.prototype.demoteLevel = function () {
    if (this.phase === "random") {
      return;
    }
    var level = M.normalizeLevel(this.minimaxDepth);
    if (level > C.MINIMAX_START_LEVEL) {
      this.minimaxDepth = M.normalizeLevel(level - C.MINIMAX_LEVEL_STEP);
    } else {
      this.phase = "random";
      this.minimaxDepth = 0;
    }
    this.phaseGamesTotal = 0;
  };

  AlgoState.prototype.toDict = function () {
    return {
      algo_id: this.algoId,
      phase: this.phase,
      minimax_depth: this.minimaxDepth,
      phase_games_total: this.phaseGamesTotal,
      random_games_total: this.randomGamesTotal,
      games_played: this.gamesPlayed,
      greedy_recent: this.greedyRecent.slice(),
      nemesis: this.nemesis.toJSON(),
      session_points: this.sessionPoints,
      dropped_after_tournament: this.droppedAfterTournament
    };
  };

  AlgoState.fromDict = function (data) {
    var state = new AlgoState(data.algo_id);
    var phase = data.phase || "random";
    if (phase === "selfplay") {
      phase = "minimax";
    }
    state.phase = phase;
    var depthRaw = data.minimax_depth;
    if (depthRaw === undefined || depthRaw === null) {
      depthRaw = data.minimaxDepth;
    }
    var depth = parseFloat(depthRaw || 0);
    if (phase === "minimax") {
      if (depth <= 0) {
        depth = C.MINIMAX_START_LEVEL;
      }
      state.minimaxDepth = M.normalizeLevel(depth);
    } else {
      state.minimaxDepth = 0;
    }
    state.phaseGamesTotal = data.phase_games_total || 0;
    state.randomGamesTotal = data.random_games_total || 0;
    state.gamesPlayed = data.games_played || 0;
    state.greedyRecent = (data.greedy_recent || []).slice(0, C.GREEDY_EVAL_GAMES);
    state.nemesis.load(data.nemesis);
    state.sessionPoints = data.session_points || 0;
    state.droppedAfterTournament = typeof data.dropped_after_tournament === "number" ?
      data.dropped_after_tournament : null;
    return state;
  };

  /* --- AgentRegistry --- */
  function AgentRegistry() {
    var v;
    for (v = 0; v < NN_VARIANT_DEPTHS.length; v += 1) {
      var depth = NN_VARIANT_DEPTHS[v];
      this["nn" + depth] = new NnAgent(NN_POLICY_LAYER_SIZES[depth]);
      this["nn_value" + depth] = new NnValueAgent(NN_VALUE_LAYER_SIZES[depth]);
      this["reinforce" + depth] = new ReinforceAgent(undefined, NN_POLICY_LAYER_SIZES[depth]);
      this["reinforce_value" + depth] = new ReinforceValueAgent(NN_VALUE_LAYER_SIZES[depth]);
    }
    this.neat = new SimpleGaNeatAgent();
    this.neat_value = new SimpleGaNeatValueAgent();
    this.qtable = new OPP.QBrain();
    this.sarsa = new SarsaBrain();
    this.menace = new OPP.MenaceBrain();
    this.genetic_menace = new GeneticMenaceAgent();
    if (!TfjsPolicyAgent || !TfjsValueAgent) {
      throw new Error("C4_TFJS.TfjsPolicyAgent and TfjsValueAgent required (load c4_js/tfjs-agent.js)");
    }
    this.tfjs = new TfjsPolicyAgent(NN_POLICY_LAYER_SIZES[2] || [78, 64, 78, 7]);
    this.tfjs_value = new TfjsValueAgent(NN_VALUE_LAYER_SIZES[2] || [77, 64, 77, 1]);
    this.tfjs_value5 = new TfjsValueAgent(
      C.TFJS_VALUE_5L_SIZES || [77, 256, 192, 160, 128, 96, 1]
    );
    this.mcts = new MctsAgent();
    this.minimax = new MinimaxTournamentAgent();
    this.states = {};
    var i;
    for (i = 0; i < TRAINING_ALGO_IDS.length; i += 1) {
      this.states[TRAINING_ALGO_IDS[i]] = new AlgoState(TRAINING_ALGO_IDS[i]);
    }
  }

  AgentRegistry.prototype.getAgent = function (algoId) {
    return this[algoId];
  };

  AgentRegistry.prototype.activeTrainingIds = function () {
    var out = [];
    var i;
    for (i = 0; i < TRAINING_ALGO_IDS.length; i += 1) {
      var aid = TRAINING_ALGO_IDS[i];
      if (this.states[aid].isActive()) {
        out.push(aid);
      }
    }
    return out;
  };

  AgentRegistry.prototype.tournamentAlgoIds = function () {
    var active = this.activeTrainingIds();
    return active.concat(["mcts", "minimax"]);
  };

  AgentRegistry.prototype.extraStats = function (algoId) {
    if (isTrainableNn(algoId)) {
      var nnAgent = this.getAgent(algoId);
      if (!nnAgent) {
        return "";
      }
      if (/^reinforce\d$/.test(algoId)) {
        return "steps=" + nnAgent.trainSteps + " loss=" + nnAgent.lastLoss.toFixed(3);
      }
      if (isTfjsAlgo(algoId) && typeof nnAgent.backend === "function") {
        return "steps=" + nnAgent.trainSteps + " backend=" + nnAgent.backend();
      }
      return "steps=" + nnAgent.trainSteps;
    }
    if (algoId === "neat") {
      return this.neat.extraStats();
    }
    if (algoId === "neat_value") {
      return this.neat_value.extraStats();
    }
    if (algoId === "qtable") {
      return "states=" + this.qtable.stateCount() +
        tabularExtraLabel(this, "qtable");
    }
    if (algoId === "sarsa") {
      return "states=" + this.sarsa.stateCount() +
        tabularExtraLabel(this, "sarsa");
    }
    if (algoId === "menace") {
      return "boxes=" + this.menace.boxCount() +
        tabularExtraLabel(this, "menace");
    }
    if (algoId === "genetic_menace") {
      return this.genetic_menace.extraStats() + tabularExtraLabel(this, "genetic_menace");
    }
    if (algoId === "mcts") {
      return this.mcts.extraStats();
    }
    if (algoId === "minimax") {
      return this.minimax.extraStats();
    }
    return "";
  };

  AgentRegistry.prototype.allStatesDict = function () {
    var out = {};
    var i;
    for (i = 0; i < TRAINING_ALGO_IDS.length; i += 1) {
      var aid = TRAINING_ALGO_IDS[i];
      out[aid] = this.states[aid].toDict();
    }
    return out;
  };

  AgentRegistry.prototype.loadStatesDict = function (data) {
    if (!data) {
      return;
    }
    data = migrateLegacyStates(data);
    var i;
    for (i = 0; i < TRAINING_ALGO_IDS.length; i += 1) {
      var aid = TRAINING_ALGO_IDS[i];
      if (data[aid]) {
        this.states[aid] = AlgoState.fromDict(data[aid]);
      }
    }
  };

  AgentRegistry.prototype.agentsDict = function () {
    return this.agentsDictForSave(null);
  };

  AgentRegistry.prototype.agentsDictForSave = function (saveOpts) {
    saveOpts = saveOpts || {};
    var tabularMax = saveOpts.tabularMaxStates;
    var qJson = tabularMax ?
      this.qtable.toJSONMaxStates(tabularMax) : this.qtable.toJSON();
    var sarsaJson = tabularMax ?
      this.sarsa.toJSONMaxStates(tabularMax) : this.sarsa.toJSON();
    var menaceJson = tabularMax ?
      this.menace.toJSONMaxBoxes(tabularMax) : this.menace.toJSON();
    var agents = {
      neat: this.neat.toDict(true),
      neat_value: this.neat_value.toDict(true),
      qtable: qJson,
      sarsa: sarsaJson,
      menace: menaceJson,
      genetic_menace: this.genetic_menace.toDict(true),
      tfjs: this.tfjs ? this.tfjs.toDict() : null,
      tfjs_value: this.tfjs_value ? this.tfjs_value.toDict() : null,
      tfjs_value5: this.tfjs_value5 ? this.tfjs_value5.toDict() : null,
      mcts: this.mcts.toDict(),
      minimax: this.minimax.toDict()
    };
    var v;
    for (v = 0; v < NN_VARIANT_DEPTHS.length; v += 1) {
      var depth = NN_VARIANT_DEPTHS[v];
      agents["nn" + depth] = this["nn" + depth].toDict();
      agents["nn_value" + depth] = this["nn_value" + depth].toDict();
      agents["reinforce" + depth] = this["reinforce" + depth].toDict();
      agents["reinforce_value" + depth] = this["reinforce_value" + depth].toDict();
    }
    return agents;
  };

  AgentRegistry.prototype.loadAgentsDict = function (data) {
    if (!data) {
      return;
    }
    data = migrateLegacyAgents(data);
    var self = this;
    var errors = [];
    function safe(name, fn) {
      try {
        fn();
      } catch (e) {
        errors.push(name + ": " + (e.message || e));
      }
    }
    var v;
    for (v = 0; v < NN_VARIANT_DEPTHS.length; v += 1) {
      var depth = NN_VARIANT_DEPTHS[v];
      safe("nn" + depth, function (d) {
        return function () { self["nn" + d] = NnAgent.fromDict(data["nn" + d]); };
      }(depth));
      safe("nn_value" + depth, function (d) {
        return function () { self["nn_value" + d] = NnValueAgent.fromDict(data["nn_value" + d]); };
      }(depth));
      safe("reinforce" + depth, function (d) {
        return function () { self["reinforce" + d] = ReinforceAgent.fromDict(data["reinforce" + d]); };
      }(depth));
      safe("reinforce_value" + depth, function (d) {
        return function () {
          self["reinforce_value" + d] = ReinforceValueAgent.fromDict(data["reinforce_value" + d]);
        };
      }(depth));
    }
    safe("neat", function () { self.neat = SimpleGaNeatAgent.fromDict(data.neat); });
    safe("neat_value", function () { self.neat_value = SimpleGaNeatValueAgent.fromDict(data.neat_value); });
    safe("qtable", function () {
      self.qtable = new OPP.QBrain();
      self.qtable.load(data.qtable);
    });
    safe("sarsa", function () { self.sarsa = SarsaBrain.fromDict(data.sarsa); });
    safe("menace", function () {
      self.menace = new OPP.MenaceBrain();
      self.menace.load(data.menace);
    });
    safe("genetic_menace", function () {
      self.genetic_menace = GeneticMenaceAgent.fromDict(data.genetic_menace);
    });
    safe("tfjs", function () {
      if (!TfjsPolicyAgent) {
        return;
      }
      self.tfjs = TfjsPolicyAgent.fromDict(data.tfjs);
    });
    safe("tfjs_value", function () {
      if (!TfjsValueAgent || !data.tfjs_value) {
        return;
      }
      self.tfjs_value = TfjsValueAgent.fromDict(data.tfjs_value);
    });
    safe("tfjs_value5", function () {
      if (!TfjsValueAgent || !data.tfjs_value5) {
        return;
      }
      self.tfjs_value5 = TfjsValueAgent.fromDict(data.tfjs_value5);
    });
    safe("mcts", function () { self.mcts = MctsAgent.fromDict(data.mcts); });
    safe("minimax", function () { self.minimax = MinimaxTournamentAgent.fromDict(data.minimax); });
    if (errors.length) {
      throw new Error(errors.join("; "));
    }
  };

  function sortKey(registry, algoId, points) {
    var state = registry.states[algoId];
    var pts = points[algoId] || 0;
    if (state.isActive()) {
      return [-pts, 0, 0];
    }
    var dropped = state.droppedAfterTournament || 0;
    return [-pts, 1, -dropped];
  }

  function trainingStatusOrder(registry, latestPoints) {
    var points = latestPoints || {};
    return TRAINING_ALGO_IDS.slice().sort(function (a, b) {
      var ka = sortKey(registry, a, points);
      var kb = sortKey(registry, b, points);
      if (ka[0] !== kb[0]) {
        return ka[0] - kb[0];
      }
      if (ka[1] !== kb[1]) {
        return ka[1] - kb[1];
      }
      return ka[2] - kb[2];
    });
  }

  function activeTrainingOrder(registry, latestPoints) {
    var ordered = trainingStatusOrder(registry, latestPoints);
    var out = [];
    var i;
    for (i = 0; i < ordered.length; i += 1) {
      if (registry.states[ordered[i]].isActive()) {
        out.push(ordered[i]);
      }
    }
    return out;
  }

  function tabularChooseMove(agent, board, player, explore) {
    return agent.chooseMove(board, player, G.boardKey, G.legalMoves, explore);
  }

  function makeSeededRng(seed) {
    return M.makeSeededRng(seed);
  }

  function nemesisLevelForState(state) {
    return state.phase === "random" ? "random" : M.normalizeLevel(state.minimaxDepth);
  }

  function gameMeta(seed, level) {
    return {
      seed: seed,
      level: OPP.normalizeNemesisLevel(level)
    };
  }

  function isTabularAlgo(algoId) {
    var ids = C.TABULAR_ALGO_IDS || ["qtable", "sarsa", "menace", "genetic_menace"];
    return ids.indexOf(algoId) >= 0;
  }

  function tabularStateCount(registry, algoId) {
    if (algoId === "qtable") {
      return registry.qtable.stateCount();
    }
    if (algoId === "sarsa") {
      return registry.sarsa.stateCount();
    }
    if (algoId === "menace") {
      return registry.menace.boxCount();
    }
    if (algoId === "genetic_menace") {
      return registry.genetic_menace.champion().boxCount();
    }
    return 0;
  }

  function tabularStateCap(algoId) {
    if (algoId === "qtable" || algoId === "sarsa") {
      return C.Q_MAX_STATES;
    }
    if (algoId === "menace" || algoId === "genetic_menace") {
      return C.MENACE_MAX_BOXES;
    }
    return 0;
  }

  function tabularBrain(registry, algoId) {
    if (algoId === "qtable") {
      return registry.qtable;
    }
    if (algoId === "sarsa") {
      return registry.sarsa;
    }
    if (algoId === "menace") {
      return registry.menace;
    }
    if (algoId === "genetic_menace") {
      return registry.genetic_menace.champion();
    }
    return null;
  }

  function tabularLruLowWatermark(cap) {
    var batch = Math.max(1, Math.floor(cap * (OPP.EVICT_FRACTION || 0.1)));
    return cap - batch;
  }

  function tabularStorageCapReached(registry, algoId, inferLegacy) {
    var brain = tabularBrain(registry, algoId);
    if (!brain) {
      return false;
    }
    if (brain.storageCapReached()) {
      return true;
    }
    if (!inferLegacy) {
      return false;
    }
    var cap = tabularStateCap(algoId);
    if (tabularStateCount(registry, algoId) >= tabularLruLowWatermark(cap)) {
      brain.lruCapReached = true;
      return true;
    }
    return false;
  }

  function tabularTrainingDone(registry, algoId) {
    if (!isTabularAlgo(algoId)) {
      return false;
    }
    var state = registry.states[algoId];
    if (!state) {
      return false;
    }
    var gamesCap = C.TABULAR_MAX_TRAINING_GAMES || 10000;
    if (state.gamesPlayed < gamesCap) {
      return false;
    }
    return tabularStorageCapReached(registry, algoId, true);
  }

  function tabularExtraLabel(registry, algoId) {
    if (tabularTrainingDone(registry, algoId)) {
      return " done";
    }
    return "";
  }

  function nemesisEligible(algoId) {
    return C.NEMESIS_ALGO_IDS.indexOf(algoId) >= 0;
  }

  function purgeIneligibleNemesis(registry) {
    var i;
    for (i = 0; i < C.TRAINING_ALGO_IDS.length; i += 1) {
      var aid = C.TRAINING_ALGO_IDS[i];
      if (!nemesisEligible(aid)) {
        registry.states[aid].nemesis.seeds = [];
        registry.states[aid].nemesis.cooldown = [];
      }
    }
  }

  function randomInt(max) {
    return Math.floor(Math.random() * max);
  }

  /* --- TrainingEngine --- */
  function TrainingEngine(registry) {
    this.registry = registry;
    this.currentAlgoIndex = 0;
    this.sliceTrainingElapsed = 0;
    this.roundTrainingElapsed = 0;
    this.trainingRoundComplete = false;
    this.sliceScale = 1;
    this.rotationIndex = 0;
    this.roundActiveIds = [];
    this.roundBotIndex = 0;
    this.roundSliceLimit = 0;
    this.minimaxCache = {};
    this.greedyActiveIds = [];
    this.greedyAlgoIndex = 0;
    this.greedyGameIndex = 0;
    this.greedyBuffer = [];
    this.latestTournamentPoints = {};
    this.beginTrainingRound();
  }

  TrainingEngine.prototype.updateTournamentRankings = function (points) {
    this.latestTournamentPoints = {};
    var k;
    for (k in points) {
      if (Object.prototype.hasOwnProperty.call(points, k)) {
        this.latestTournamentPoints[k] = points[k];
      }
    }
  };

  TrainingEngine.prototype.orderedActiveIds = function () {
    var ordered = activeTrainingOrder(this.registry, this.latestTournamentPoints);
    return ordered.filter(function (algoId) {
      return !tabularTrainingDone(this.registry, algoId);
    }.bind(this));
  };

  TrainingEngine.prototype.trainingEligibleCount = function () {
    return this.orderedActiveIds().length;
  };

  TrainingEngine.prototype.beginTrainingRound = function () {
    this.roundActiveIds = this.orderedActiveIds();
    this.roundBotIndex = 0;
    this.sliceTrainingElapsed = 0;
    this.roundTrainingElapsed = 0;
    this.trainingRoundComplete = false;
    var n = this.roundActiveIds.length;
    this.roundSliceLimit = n > 0 ?
      (C.TOURNAMENT_INTERVAL || 30) * this.sliceScale / n : 0;
    if (n) {
      this.currentAlgoIndex = TRAINING_ALGO_IDS.indexOf(this.roundActiveIds[0]);
    }
  };

  TrainingEngine.prototype.roundCurrentAlgoId = function () {
    if (!this.roundActiveIds.length || this.roundBotIndex >= this.roundActiveIds.length) {
      return this.currentAlgoId();
    }
    return this.roundActiveIds[this.roundBotIndex];
  };

  TrainingEngine.prototype.adjustSliceScaleFromRound = function (actualSeconds) {
    this.sliceScale = adjustTrainingSliceScale(this.sliceScale, actualSeconds);
  };

  TrainingEngine.prototype.finishCurrentSlice = function () {
    this.roundBotIndex += 1;
    this.sliceTrainingElapsed = 0;
    if (this.roundBotIndex >= this.roundActiveIds.length) {
      this.trainingRoundComplete = true;
      this.rotationIndex += 1;
    } else {
      this.currentAlgoIndex = TRAINING_ALGO_IDS.indexOf(this.roundActiveIds[this.roundBotIndex]);
    }
  };

  TrainingEngine.prototype.currentAlgoId = function () {
    return TRAINING_ALGO_IDS[this.currentAlgoIndex];
  };

  TrainingEngine.prototype.ensureCurrentActive = function () {
    var active = this.orderedActiveIds();
    if (!active.length) {
      return;
    }
    var current = this.currentAlgoId();
    if (active.indexOf(current) < 0) {
      this.currentAlgoIndex = TRAINING_ALGO_IDS.indexOf(active[0]);
      this.sliceTrainingElapsed = 0;
    }
  };

  TrainingEngine.prototype.resetSlice = function () {
    this.sliceTrainingElapsed = 0;
  };

  TrainingEngine.prototype.sliceElapsed = function () {
    return this.sliceTrainingElapsed;
  };

  TrainingEngine.prototype.resetRoundTrainingTimer = function () {
    this.beginTrainingRound();
  };

  TrainingEngine.prototype.shouldStartTournament = function () {
    if (!this.roundActiveIds.length) {
      return this.roundTrainingElapsed > 0;
    }
    return this.trainingRoundComplete;
  };

  TrainingEngine.prototype.estimatedRotationRemaining = function () {
    var botsLeft = this.roundActiveIds.length - this.roundBotIndex;
    if (botsLeft <= 0) {
      return 0;
    }
    var rem = botsLeft * this.roundSliceLimit - this.sliceTrainingElapsed;
    return rem < 0 ? 0 : rem;
  };

  TrainingEngine.prototype.recordTrainingDuration = function (seconds) {
    this.sliceTrainingElapsed += seconds;
    this.roundTrainingElapsed += seconds;
  };

  TrainingEngine.prototype.advanceAlgo = function () {
    var active = this.orderedActiveIds();
    if (!active.length) {
      return;
    }
    var current = this.currentAlgoId();
    var idx = active.indexOf(current);
    if (idx < 0) {
      idx = -1;
    }
    var nextIdx = (idx + 1) % active.length;
    var nextId = active[nextIdx];
    this.currentAlgoIndex = TRAINING_ALGO_IDS.indexOf(nextId);
    if (nextIdx === 0) {
      this.rotationIndex += 1;
    }
    this.resetSlice();
  };

  TrainingEngine.prototype.fractionalMinimax = function (level) {
    var key = M.normalizeLevel(level);
    if (!this.minimaxCache[key]) {
      this.minimaxCache[key] = new M.FractionalMinimaxOpponent(key);
    }
    return this.minimaxCache[key];
  };

  TrainingEngine.prototype.runTrainingUntil = function (deadline) {
    while (performance.now() < deadline) {
      this.runOneStep();
      if (this.shouldStartTournament()) {
        if (this.trainingRoundComplete) {
          this.adjustSliceScaleFromRound(this.roundTrainingElapsed);
        }
        return false;
      }
    }
    return true;
  };

  TrainingEngine.prototype.runGreedyUntil = function (deadline) {
    while (performance.now() < deadline) {
      if (!this.playOneGreedyEvalGame()) {
        return false;
      }
    }
    return true;
  };

  TrainingEngine.prototype.runOneStep = function () {
    return tracedCall("runOneStep", null, function () {
      if (!this.roundActiveIds.length) {
        this.trainingRoundComplete = true;
        return;
      }
      if (this.sliceTrainingElapsed >= this.roundSliceLimit) {
        this.finishCurrentSlice();
        return;
      }
      var start = performance.now();
      var algoId = this.roundCurrentAlgoId();
      this.currentAlgoIndex = TRAINING_ALGO_IDS.indexOf(algoId);
      tracePush("runOneStep.algo", { algo: algoId });
      try {
        var state = this.registry.states[algoId];
        if (!state.isActive() || tabularTrainingDone(this.registry, algoId)) {
          this.finishCurrentSlice();
          return;
        }
        if (this.trainNemesisDrill(algoId)) {
          this.recordTrainingDuration((performance.now() - start) / 1000);
          return;
        }
        if (algoId === "neat") {
          this.neatOneGeneration(state);
        } else if (algoId === "neat_value") {
          this.neatValueOneGeneration(state);
        } else if (algoId === "genetic_menace") {
          this.geneticOneGeneration(state);
        } else if (state.phase === "random") {
          this.trainVsRandom(algoId, true, true, randomInt(2147483647));
        } else {
          this.trainVsMinimax(algoId, state.minimaxDepth, true, true, randomInt(2147483647));
        }
        this.recordTrainingDuration((performance.now() - start) / 1000);
      } finally {
        tracePop();
      }
    }.bind(this));
  };

  TrainingEngine.prototype.startGreedyEvalRound = function () {
    this.greedyActiveIds = this.orderedActiveIds();
    this.greedyAlgoIndex = 0;
    this.greedyGameIndex = 0;
    this.greedyBuffer = [];
    this.greedyEvalSeeds = [];
    var i;
    for (i = 0; i < C.GREEDY_EVAL_GAMES; i += 1) {
      this.greedyEvalSeeds.push(randomInt(2147483647));
    }
  };

  TrainingEngine.prototype.playOneGreedyEvalGame = function () {
    if (this.greedyAlgoIndex >= this.greedyActiveIds.length) {
      return false;
    }
    var algoId = this.greedyActiveIds[this.greedyAlgoIndex];
    this.greedyBuffer.push(this.playOneGreedyGame(algoId));
    this.greedyGameIndex += 1;
    if (this.greedyGameIndex < C.GREEDY_EVAL_GAMES) {
      return true;
    }
    var state = this.registry.states[algoId];
    state.greedyRecent = this.greedyBuffer.slice(0, C.GREEDY_EVAL_GAMES);
    state.maybeGraduate();
    this.greedyBuffer = [];
    this.greedyGameIndex = 0;
    this.greedyAlgoIndex += 1;
    return this.greedyAlgoIndex < this.greedyActiveIds.length;
  };

  TrainingEngine.prototype.greedyEvalInProgress = function () {
    return this.greedyActiveIds.length > 0 &&
      this.greedyAlgoIndex < this.greedyActiveIds.length;
  };

  TrainingEngine.prototype.greedyEvalProgress = function () {
    if (!this.greedyEvalInProgress()) {
      return null;
    }
    var algoId = this.greedyActiveIds[this.greedyAlgoIndex];
    return (this.greedyAlgoIndex + 1) + "/" + this.greedyActiveIds.length + " " +
      displayName(algoId) + " " + this.greedyGameIndex + "/" + C.GREEDY_EVAL_GAMES;
  };

  TrainingEngine.prototype.noteGreedyLoss = function (state, result, seed, level, algoId) {
    if (!nemesisEligible(algoId)) {
      return;
    }
    if (result === "loss" && typeof seed === "number") {
      state.nemesis.addNemesis(seed, level);
    }
  };

  TrainingEngine.prototype.playOneGreedyGame = function (algoId) {
    var state = this.registry.states[algoId];
    var seed = this.greedyEvalSeeds ? this.greedyEvalSeeds[this.greedyGameIndex] : null;
    var level = nemesisLevelForState(state);
    var result;
    if (algoId === "neat") {
      var genome = this.registry.neat.bestGenomeForPlay();
      if (state.phase === "random") {
        result = this.playNeatGenomeVsRandom(genome, false, seed);
        this.noteGreedyLoss(state, result, seed, level, algoId);
        return result;
      }
      result = this.playNeatGenomeVsMinimax(genome, state.minimaxDepth, false, seed);
      this.noteGreedyLoss(state, result, seed, level, algoId);
      return result;
    }
    if (algoId === "neat_value") {
      var valueGenome = this.registry.neat_value.bestGenomeForPlay();
      if (state.phase === "random") {
        result = this.playNeatValueGenomeVsRandom(valueGenome, false, seed);
        this.noteGreedyLoss(state, result, seed, level, algoId);
        return result;
      }
      result = this.playNeatValueGenomeVsMinimax(valueGenome, state.minimaxDepth, false, seed);
      this.noteGreedyLoss(state, result, seed, level, algoId);
      return result;
    }
    if (algoId === "genetic_menace") {
      var brain = this.registry.genetic_menace.champion();
      if (state.phase === "random") {
        result = this.playMenaceVsRandom(brain, false, false, seed);
        this.noteGreedyLoss(state, result, seed, level, algoId);
        return result;
      }
      result = this.playMenaceVsMinimax(brain, state.minimaxDepth, false, false, seed);
      this.noteGreedyLoss(state, result, seed, level, algoId);
      return result;
    }
    if (state.phase === "random") {
      result = this.trainVsRandom(algoId, false, false, seed);
      this.noteGreedyLoss(state, result, seed, level, algoId);
      return result;
    }
    result = this.trainVsMinimax(algoId, state.minimaxDepth, false, false, seed);
    this.noteGreedyLoss(state, result, seed, level, algoId);
    return result;
  };

  TrainingEngine.prototype.neatFitness = function (state) {
    var self = this;
    var games = gaFitnessGames();
    return function (genome, index) {
      var wins = 0;
      var g;
      for (g = 0; g < games; g += 1) {
        wins += fitnessPoints(self.playNeatGenomeVsPeer(genome, index, true));
      }
      return wins / games;
    };
  };

  TrainingEngine.prototype.neatOneGeneration = function (state) {
    this.registry.neat.evolveOneGeneration(this.neatFitness(state));
    var neatGames = C.NEAT_POP_SIZE * gaFitnessGames();
    state.gamesPlayed += neatGames;
    state.phaseGamesTotal += neatGames;
    if (state.phase === "random") {
      state.randomGamesTotal += neatGames;
    }
  };

  TrainingEngine.prototype.neatValueFitness = function (state) {
    var self = this;
    var games = gaFitnessGames();
    return function (genome, index) {
      var wins = 0;
      var g;
      for (g = 0; g < games; g += 1) {
        wins += fitnessPoints(self.playNeatValueGenomeVsPeer(genome, index, true));
      }
      return wins / games;
    };
  };

  TrainingEngine.prototype.neatValueOneGeneration = function (state) {
    this.registry.neat_value.evolveOneGeneration(this.neatValueFitness(state));
    var neatGames = C.NEAT_POP_SIZE * gaFitnessGames();
    state.gamesPlayed += neatGames;
    state.phaseGamesTotal += neatGames;
    if (state.phase === "random") {
      state.randomGamesTotal += neatGames;
    }
  };

  TrainingEngine.prototype.geneticOneGeneration = function (state) {
    var agent = this.registry.genetic_menace;
    var games = gaFitnessGames();
    var fitnessScores = [];
    var i;
    var g;
    for (i = 0; i < agent.population.length; i += 1) {
      var individual = agent.population[i];
      var wins = 0;
      for (g = 0; g < games; g += 1) {
        wins += fitnessPoints(this.playMenaceVsPeer(individual, i, true));
      }
      fitnessScores.push(wins / games);
    }
    agent.evolve(fitnessScores);
    var totalGames = fitnessScores.length * games;
    state.gamesPlayed += totalGames;
    state.phaseGamesTotal += totalGames;
    if (state.phase === "random") {
      state.randomGamesTotal += totalGames;
    }
  };

  TrainingEngine.prototype.recordTrainingGame = function (state, explore, result, lossMeta, algoId) {
    state.gamesPlayed += 1;
    state.phaseGamesTotal += 1;
    state.nemesis.tickEpisode();
    if (state.phase === "random") {
      state.randomGamesTotal += 1;
    }
    if (explore && result === "loss" && lossMeta && typeof lossMeta.seed === "number" &&
        nemesisEligible(algoId)) {
      this.handleExploreLoss(algoId, lossMeta.seed, lossMeta.level);
    }
  };

  TrainingEngine.prototype.trainNemesisDrill = function (algoId) {
    return tracedCall("trainNemesisDrill", { algo: algoId }, function () {
      if (!nemesisEligible(algoId)) {
        return false;
      }
      var state = this.registry.states[algoId];
      var entry = state.nemesis.pickEligibleEntry();
      if (!entry) {
        return false;
      }
      var result = this.playNemesisDrillGame(algoId, entry);
      this.handleNemesisOutcome(algoId, entry, result);
      if (result === "loss") {
        this.handleExploreLoss(algoId, entry.seed, entry.level);
      }
      return true;
    }.bind(this));
  };

  TrainingEngine.prototype.playSeededEvalGame = function (algoId, seed, level, explore, train) {
    return tracedCall("playSeededEvalGame", {
      algo: algoId, seed: seed, level: level, explore: explore, train: train
    }, function () {
      level = OPP.normalizeNemesisLevel(level);
      if (level === "random") {
        return this.playRandomEvalGame(algoId, seed, explore, train);
      }
      if (algoId === "neat") {
        return this.playNeatGenomeVsMinimax(
          this.registry.neat.bestGenomeForPlay(), level, explore, seed, train
        );
      }
      if (algoId === "neat_value") {
        return this.playNeatValueGenomeVsMinimax(
          this.registry.neat_value.bestGenomeForPlay(), level, explore, seed, train
        );
      }
      if (algoId === "genetic_menace") {
        return this.playMenaceVsMinimax(
          this.registry.genetic_menace.champion(), level, explore, train, seed,
          train ? "genetic_menace" : null
        );
      }
      return this.trainVsMinimax(algoId, level, explore, train, seed, false);
    }.bind(this));
  };

  TrainingEngine.prototype.playRandomEvalGame = function (algoId, seed, explore, train) {
    if (algoId === "neat") {
      return this.playNeatGenomeVsRandom(
        this.registry.neat.bestGenomeForPlay(), explore, seed, train
      );
    }
    if (algoId === "neat_value") {
      return this.playNeatValueGenomeVsRandom(
        this.registry.neat_value.bestGenomeForPlay(), explore, seed, train
      );
    }
    if (algoId === "genetic_menace") {
      return this.playMenaceVsRandom(
        this.registry.genetic_menace.champion(), explore, train, seed,
        train ? "genetic_menace" : null
      );
    }
    return this.trainVsRandom(algoId, explore, train, seed, false);
  };

  TrainingEngine.prototype.playNemesisDrillGame = function (algoId, entry) {
    /* Exploit only: practice the hard seed without random explore noise. */
    return this.playSeededEvalGame(algoId, entry.seed, entry.level, false, true);
  };

  TrainingEngine.prototype.handleNemesisOutcome = function (algoId, entry, result) {
    var tracker = this.registry.states[algoId].nemesis;
    if (tracker.findEntryIndex(entry.seed, entry.level) < 0) {
      return;
    }
    if (result === "win" || result === "draw") {
      tracker.recordNemesisSuccess(entry.seed, entry.level);
    } else if (result === "loss") {
      tracker.recordNemesisLoss(entry.seed, entry.level);
    }
  };

  TrainingEngine.prototype.trainVsRandom = function (algoId, explore, train, seed, trackNemesis) {
    if (typeof trackNemesis === "undefined") {
      trackNemesis = true;
    }
    var rng = seed !== null && seed !== undefined ? makeSeededRng(seed) : Math.random.bind(Math);
    var learnerMark = rng() < 0.5 ? X : O;
    var meta = typeof seed === "number" ? gameMeta(seed, "random") : null;
    return this.trainVsOpponent(algoId, learnerMark, function (board) {
      var moves = G.legalMoves(board);
      return moves[Math.floor(rng() * moves.length)];
    }, explore, train, meta, trackNemesis);
  };

  TrainingEngine.prototype.trainVsMinimax = function (algoId, level, explore, train, seed, trackNemesis) {
    if (typeof trackNemesis === "undefined") {
      trackNemesis = true;
    }
    var normLevel = M.normalizeLevel(level);
    var rng = typeof seed === "number" ? makeSeededRng(seed) : Math.random.bind(Math);
    var learnerMark = rng() < 0.5 ? X : O;
    var opponent = new M.FractionalMinimaxOpponent(normLevel, rng);
    var meta = typeof seed === "number" ? gameMeta(seed, normLevel) : null;
    return this.trainVsOpponent(algoId, learnerMark, function (board, player) {
      return opponent.chooseMove(board, player);
    }, explore, train, meta, trackNemesis);
  };

  TrainingEngine.prototype.trainVsOpponent = function (
    algoId, learnerMark, opponentMoveFn, explore, train, lossMeta, trackNemesis
  ) {
    return tracedCall("trainVsOpponent", {
      algo: algoId, explore: explore, train: train, trackNemesis: trackNemesis
    }, function () {
      var board = G.emptyBoard();
      var current = X;
      var trajectory = [];
      var state = this.registry.states[algoId];
      while (true) {
        var moves = G.legalMoves(board);
        if (!moves.length) {
          break;
        }
        var move;
        if (current === learnerMark) {
          move = this.pickMove(algoId, board, current, explore);
          if (train) {
            trajectory.push(this.buildStep(algoId, board, current, move));
          }
        } else {
          move = opponentMoveFn(board, current);
        }
        board = G.applyMove(board, current, move);
        if (G.findWinner(board) || G.isDraw(board)) {
          break;
        }
        current = G.other(current);
      }
      var winner = G.findWinner(board);
      if (train && trajectory.length) {
        this.learn(algoId, trajectory, learnerMark, winner);
      }
      var result = G.resultForPlayer(winner, learnerMark);
      if (train) {
        this.recordTrainingGame(state, explore, result, trackNemesis ? lossMeta : null, algoId);
      }
      return result;
    }.bind(this));
  };

  function playMenaceBrainVsRandom(brain, explore, train, seed) {
    var rng = typeof seed === "number" ? makeSeededRng(seed) : Math.random.bind(Math);
    var learnerMark = rng() < 0.5 ? X : O;
    var board = G.emptyBoard();
    var current = X;
    var trajectory = [];
    while (true) {
      var moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      var move;
      if (current === learnerMark) {
        move = brain.chooseMove(board, current, G.boardKey, G.legalMoves, explore);
        if (train) {
          trajectory.push(buildTabularStep(board, current, move));
        }
      } else {
        move = moves[Math.floor(rng() * moves.length)];
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    var winner = G.findWinner(board);
    if (train && trajectory.length) {
      brain.learnFromTrajectory(trajectory, winner);
    }
    return G.resultForPlayer(winner, learnerMark);
  }

  TrainingEngine.prototype.playMenaceVsRandom = function (brain, explore, train, seed, recordAlgoId) {
    var result = playMenaceBrainVsRandom(brain, explore, train, seed);
    if (train && recordAlgoId) {
      var meta = typeof seed === "number" ? gameMeta(seed, "random") : null;
      this.recordTrainingGame(this.registry.states[recordAlgoId], explore, result, meta, recordAlgoId);
    }
    return result;
  };

  TrainingEngine.prototype.playMenaceVsMinimax = function (brain, level, explore, train, seed, recordAlgoId) {
    var normLevel = M.normalizeLevel(level);
    var rng = typeof seed === "number" ? makeSeededRng(seed) : Math.random.bind(Math);
    var learnerMark = rng() < 0.5 ? X : O;
    var opponent = new M.FractionalMinimaxOpponent(normLevel, rng);
    var board = G.emptyBoard();
    var current = X;
    var trajectory = [];
    var moves;
    var move;
    var winner;
    while (true) {
      moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      if (current === learnerMark) {
        move = brain.chooseMove(board, current, G.boardKey, G.legalMoves, explore);
        if (train) {
          trajectory.push(buildTabularStep(board, current, move));
        }
      } else {
        move = opponent.chooseMove(board, current);
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    winner = G.findWinner(board);
    if (train && trajectory.length) {
      brain.learnFromTrajectory(trajectory, winner);
    }
    var result = G.resultForPlayer(winner, learnerMark);
    if (train && recordAlgoId) {
      var meta = typeof seed === "number" ? gameMeta(seed, normLevel) : null;
      this.recordTrainingGame(this.registry.states[recordAlgoId], explore, result, meta, recordAlgoId);
    }
    return result;
  };

  TrainingEngine.prototype.playNeatGenomeVsRandom = function (genome, explore, seed, countTraining) {
    var rng = typeof seed === "number" ? makeSeededRng(seed) : Math.random.bind(Math);
    var learnerMark = rng() < 0.5 ? X : O;
    var board = G.emptyBoard();
    var current = X;
    var moves;
    var move;
    while (true) {
      moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      if (current === learnerMark) {
        move = this.registry.neat.chooseMoveWithGenome(
          genome, board, current, explore, this.registry.states.neat.gamesPlayed
        );
      } else {
        move = moves[Math.floor(rng() * moves.length)];
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    var result = G.resultForPlayer(G.findWinner(board), learnerMark);
    if (countTraining) {
      var meta = typeof seed === "number" ? gameMeta(seed, "random") : null;
      this.recordTrainingGame(
        this.registry.states.neat, explore, result, this._handlingExploreLoss ? null : meta, "neat"
      );
    }
    return result;
  };

  TrainingEngine.prototype.playNeatGenomeVsMinimax = function (genome, level, explore, seed, countTraining) {
    var normLevel = M.normalizeLevel(level);
    var rng = typeof seed === "number" ? makeSeededRng(seed) : Math.random.bind(Math);
    var learnerMark = rng() < 0.5 ? X : O;
    var opponent = new M.FractionalMinimaxOpponent(normLevel, rng);
    var board = G.emptyBoard();
    var current = X;
    var moves;
    var move;
    while (true) {
      moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      if (current === learnerMark) {
        move = this.registry.neat.chooseMoveWithGenome(
          genome, board, current, explore, this.registry.states.neat.gamesPlayed
        );
      } else {
        move = opponent.chooseMove(board, current);
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    var result = G.resultForPlayer(G.findWinner(board), learnerMark);
    if (countTraining) {
      var meta = typeof seed === "number" ? gameMeta(seed, normLevel) : null;
      this.recordTrainingGame(
        this.registry.states.neat, explore, result, this._handlingExploreLoss ? null : meta, "neat"
      );
    }
    return result;
  };

  TrainingEngine.prototype.playNeatValueGenomeVsRandom = function (genome, explore, seed, countTraining) {
    var rng = typeof seed === "number" ? makeSeededRng(seed) : Math.random.bind(Math);
    var learnerMark = rng() < 0.5 ? X : O;
    var board = G.emptyBoard();
    var current = X;
    var moves;
    var move;
    while (true) {
      moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      if (current === learnerMark) {
        move = this.registry.neat_value.chooseMoveWithGenome(
          genome, board, current, explore, this.registry.states.neat_value.gamesPlayed
        );
      } else {
        move = moves[Math.floor(rng() * moves.length)];
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    var result = G.resultForPlayer(G.findWinner(board), learnerMark);
    if (countTraining) {
      var meta = typeof seed === "number" ? gameMeta(seed, "random") : null;
      this.recordTrainingGame(
        this.registry.states.neat_value, explore, result, this._handlingExploreLoss ? null : meta, "neat_value"
      );
    }
    return result;
  };

  TrainingEngine.prototype.playNeatValueGenomeVsMinimax = function (genome, level, explore, seed, countTraining) {
    var normLevel = M.normalizeLevel(level);
    var rng = typeof seed === "number" ? makeSeededRng(seed) : Math.random.bind(Math);
    var learnerMark = rng() < 0.5 ? X : O;
    var opponent = new M.FractionalMinimaxOpponent(normLevel, rng);
    var board = G.emptyBoard();
    var current = X;
    var moves;
    var move;
    while (true) {
      moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      if (current === learnerMark) {
        move = this.registry.neat_value.chooseMoveWithGenome(
          genome, board, current, explore, this.registry.states.neat_value.gamesPlayed
        );
      } else {
        move = opponent.chooseMove(board, current);
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    var result = G.resultForPlayer(G.findWinner(board), learnerMark);
    if (countTraining) {
      var meta = typeof seed === "number" ? gameMeta(seed, normLevel) : null;
      this.recordTrainingGame(
        this.registry.states.neat_value, explore, result, this._handlingExploreLoss ? null : meta, "neat_value"
      );
    }
    return result;
  };

  TrainingEngine.prototype.playNeatGenomeVsPeer = function (genome, selfIndex, explore) {
    var pop = this.registry.neat.population;
    var peerIdx = pickPeerIndex(pop.length, selfIndex);
    if (peerIdx === selfIndex) {
      return this.playNeatGenomeVsRandom(genome, explore);
    }
    var peer = pop[peerIdx];
    var learnerMark = Math.random() < 0.5 ? X : O;
    var board = G.emptyBoard();
    var current = X;
    var gamesPlayed = this.registry.states.neat.gamesPlayed;
    var moves;
    var move;
    while (true) {
      moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      if (current === learnerMark) {
        move = this.registry.neat.chooseMoveWithGenome(
          genome, board, current, explore, gamesPlayed
        );
      } else {
        move = this.registry.neat.chooseMoveWithGenome(
          peer, board, current, explore, gamesPlayed
        );
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    return G.resultForPlayer(G.findWinner(board), learnerMark);
  };

  TrainingEngine.prototype.playNeatValueGenomeVsPeer = function (genome, selfIndex, explore) {
    var pop = this.registry.neat_value.population;
    var peerIdx = pickPeerIndex(pop.length, selfIndex);
    if (peerIdx === selfIndex) {
      return this.playNeatValueGenomeVsRandom(genome, explore);
    }
    var peer = pop[peerIdx];
    var learnerMark = Math.random() < 0.5 ? X : O;
    var board = G.emptyBoard();
    var current = X;
    var gamesPlayed = this.registry.states.neat_value.gamesPlayed;
    var moves;
    var move;
    while (true) {
      moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      if (current === learnerMark) {
        move = this.registry.neat_value.chooseMoveWithGenome(
          genome, board, current, explore, gamesPlayed
        );
      } else {
        move = this.registry.neat_value.chooseMoveWithGenome(
          peer, board, current, explore, gamesPlayed
        );
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    return G.resultForPlayer(G.findWinner(board), learnerMark);
  };

  TrainingEngine.prototype.playMenaceVsPeer = function (brain, selfIndex, explore) {
    var pop = this.registry.genetic_menace.population;
    var peerIdx = pickPeerIndex(pop.length, selfIndex);
    if (peerIdx === selfIndex) {
      return this.playMenaceVsRandom(brain, explore, false);
    }
    var peer = pop[peerIdx];
    var learnerMark = Math.random() < 0.5 ? X : O;
    var board = G.emptyBoard();
    var current = X;
    var moves;
    var move;
    while (true) {
      moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      if (current === learnerMark) {
        move = brain.chooseMove(board, current, G.boardKey, G.legalMoves, explore);
      } else {
        move = peer.chooseMove(board, current, G.boardKey, G.legalMoves, explore);
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    return G.resultForPlayer(G.findWinner(board), learnerMark);
  };

  TrainingEngine.prototype.handleExploreLoss = function (algoId, seed, level) {
    if (this._handlingExploreLoss) {
      tracePush("handleExploreLoss.blocked", { algo: algoId, seed: seed, level: level });
      tracePop();
      return;
    }
    return tracedCall("handleExploreLoss", { algo: algoId, seed: seed, level: level }, function () {
      this._handlingExploreLoss = true;
      try {
        var state = this.registry.states[algoId];
        level = OPP.normalizeNemesisLevel(level);
        var verify = this.playSeededEvalGame(algoId, seed, level, false, false);
        if (verify !== "loss") {
          return;
        }
        var tracker = state.nemesis;
        tracker.addNemesis(seed, level);
        var i;
        for (i = 0; i < LOSS_REPLAY_COUNT; i += 1) {
          tracePush("handleExploreLoss.replay", { algo: algoId, replay: i + 1 });
          try {
            var outcome = this.playSeededEvalGame(algoId, seed, level, false, true);
            if (outcome === "loss") {
              tracker.addNemesis(seed, level);
              return;
            }
          } finally {
            tracePop();
          }
        }
        tracker.resetNemesisSuccesses(seed, level);
      } finally {
        this._handlingExploreLoss = false;
      }
    }.bind(this));
  };

  TrainingEngine.prototype.pickMove = function (algoId, board, player, explore) {
    var agent = this.registry.getAgent(algoId);
    var gamesPlayed = (this.registry.states[algoId] && this.registry.states[algoId].gamesPlayed) || 0;
    if (algoId === "qtable" || algoId === "sarsa" || algoId === "menace") {
      return tabularChooseMove(agent, board, player, explore);
    }
    return agent.chooseMove(board, player, explore, gamesPlayed);
  };

  TrainingEngine.prototype.buildStep = function (algoId, board, player, move) {
    var agent = this.registry.getAgent(algoId);
    if (isTrainableNn(algoId)) {
      return agent.buildStep(board, player, move);
    }
    return buildTabularStep(board, player, move);
  };

  TrainingEngine.prototype.learn = function (algoId, trajectory, mark, winner) {
    tracePush("learn", { algo: algoId, steps: trajectory.length });
    try {
      var agent = this.registry.getAgent(algoId);
      if (algoId === "qtable" || algoId === "sarsa") {
        agent.learnFromTrajectory(trajectory, mark, winner);
      } else if (algoId === "menace") {
        agent.learnFromTrajectory(trajectory, winner);
      } else if (isTrainableNn(algoId)) {
        agent.learnFromTrajectory(trajectory, mark, winner);
      }
    } finally {
      tracePop();
    }
  };

  function createTournamentResult() {
    var points = {};
    var wins = {};
    var draws = {};
    var losses = {};
    var lossTo = {};
    var i;
    for (i = 0; i < ALL_ALGO_IDS.length; i += 1) {
      var aid = ALL_ALGO_IDS[i];
      points[aid] = 0;
      wins[aid] = 0;
      draws[aid] = 0;
      losses[aid] = 0;
      lossTo[aid] = {};
    }
    return {
      points: points,
      wins: wins,
      draws: draws,
      losses: losses,
      lossTo: lossTo,
      gamesPlayed: 0,
      ranked: function () {
        var items = [];
        for (var k in this.points) {
          if (Object.prototype.hasOwnProperty.call(this.points, k)) {
            items.push([k, this.points[k]]);
          }
        }
        items.sort(function (a, b) { return b[1] - a[1]; });
        return items;
      }
    };
  }

  function mergeResultField(target, source) {
    if (!source) {
      return;
    }
    var k;
    for (k in source) {
      if (Object.prototype.hasOwnProperty.call(source, k)) {
        target[k] = source[k];
      }
    }
  }

  function tournamentResultRanked(result) {
    if (!result) {
      return [];
    }
    if (typeof result.ranked === "function") {
      return result.ranked();
    }
    if (Array.isArray(result.ranked)) {
      return result.ranked.slice();
    }
    var items = [];
    var pts = result.points || {};
    for (var k in pts) {
      if (Object.prototype.hasOwnProperty.call(pts, k)) {
        items.push([k, pts[k]]);
      }
    }
    items.sort(function (a, b) { return b[1] - a[1]; });
    return items;
  }

  function hydrateTournamentResult(raw) {
    if (!raw || typeof raw !== "object" || !raw.points) {
      return null;
    }
    var result = createTournamentResult();
    mergeResultField(result.points, raw.points);
    mergeResultField(result.wins, raw.wins);
    mergeResultField(result.draws, raw.draws);
    mergeResultField(result.losses, raw.losses);
    if (raw.lossTo) {
      var aid;
      for (aid in raw.lossTo) {
        if (Object.prototype.hasOwnProperty.call(raw.lossTo, aid)) {
          result.lossTo[aid] = {};
          mergeResultField(result.lossTo[aid], raw.lossTo[aid]);
        }
      }
    }
    result.gamesPlayed = typeof raw.gamesPlayed === "number" ? raw.gamesPlayed : 0;
    return result;
  }

  function tournamentPairs(algoIds) {
    var pairs = [];
    var i;
    for (i = 0; i < algoIds.length; i += 1) {
      var j;
      for (j = i + 1; j < algoIds.length; j += 1) {
        pairs.push([algoIds[i], algoIds[j]]);
      }
    }
    return pairs;
  }

  function tournamentGameCount(algoIds) {
    var n = algoIds.length;
    return n >= 2 ? n * (n - 1) : 0;
  }

  function clampTournamentMoveBudget(ms) {
    var min = C.TOURNAMENT_MOVE_BUDGET_MIN_MS || 0.25;
    var max = C.TOURNAMENT_MOVE_BUDGET_MAX_MS || 200;
    if (ms < min) {
      return min;
    }
    if (ms > max) {
      return max;
    }
    return ms;
  }

  function computeInitialTournamentMoveBudgetMs(expectedGames) {
    var target = C.TOURNAMENT_TARGET_MS || 10000;
    var maxPlies = C.TOURNAMENT_MAX_PLIES_PER_GAME || C.CELLS;
    var totalPlies = Math.max(1, expectedGames * maxPlies);
    return clampTournamentMoveBudget(target / totalPlies);
  }

  function adjustTournamentMoveBudget(currentBudget, actualMs) {
    var target = C.TOURNAMENT_TARGET_MS || 10000;
    if (!actualMs || actualMs <= 0) {
      return currentBudget;
    }
    if (!currentBudget || currentBudget <= 0) {
      return clampTournamentMoveBudget(target / 1000);
    }
    var blend = typeof C.TOURNAMENT_BUDGET_ADJUST_BLEND === "number" ?
      C.TOURNAMENT_BUDGET_ADJUST_BLEND : 0.5;
    var corrected = currentBudget * (target / actualMs);
    return clampTournamentMoveBudget(currentBudget * (1 - blend) + corrected * blend);
  }

  function clampTrainingSliceScale(scale) {
    var min = typeof C.TRAINING_SLICE_SCALE_MIN === "number" ? C.TRAINING_SLICE_SCALE_MIN : 0.25;
    var max = typeof C.TRAINING_SLICE_SCALE_MAX === "number" ? C.TRAINING_SLICE_SCALE_MAX : 4;
    if (scale < min) {
      return min;
    }
    if (scale > max) {
      return max;
    }
    return scale;
  }

  function adjustTrainingSliceScale(currentScale, actualSeconds) {
    var target = C.TOURNAMENT_INTERVAL || 30;
    if (!actualSeconds || actualSeconds <= 0) {
      return currentScale;
    }
    if (!currentScale || currentScale <= 0) {
      return 1;
    }
    var corrected = currentScale * (target / actualSeconds);
    return clampTrainingSliceScale((currentScale + corrected) / 2);
  }

  function isValueAgent(algoId) {
    return VALUE_AGENT_IDS.indexOf(algoId) >= 0;
  }

  function isNnTimedSearchBot(algoId) {
    return isTrainableNn(algoId) || algoId === "neat" || algoId === "neat_value";
  }

  function playAlgoHasTimedSearch(algoId) {
    return algoId === "mcts" || algoId === "minimax" || isNnTimedSearchBot(algoId);
  }

  function pickPlayMove(registry, algoId, board, player, budgetSec) {
    var agent = registry.getAgent(algoId);
    if (!agent) {
      throw new Error("Unknown play algorithm: " + algoId);
    }
    var budgetMs = playAlgoHasTimedSearch(algoId) ?
      Math.max(0, Number(budgetSec) || 0) * 1000 : 0;
    if (algoId === "mcts") {
      return agent.chooseMove(board, player, false, budgetMs);
    }
    if (algoId === "minimax") {
      return agent.chooseMove(board, player, budgetMs);
    }
    if (isNnTimedSearchBot(algoId)) {
      return agent.chooseMoveTimed(board, player, budgetMs);
    }
    if (algoId === "qtable" || algoId === "sarsa" || algoId === "menace") {
      return tabularChooseMove(agent, board, player, false);
    }
    if (algoId === "genetic_menace") {
      return agent.chooseMove(board, player, false);
    }
    if (typeof agent.chooseMoveTimed === "function" && budgetMs > 0) {
      return agent.chooseMoveTimed(board, player, budgetMs);
    }
    return agent.chooseMove(board, player, false);
  }

  /* --- TournamentRunner --- */
  function TournamentRunner(registry) {
    this.registry = registry;
    this.pendingGames = [];
    this.currentResult = null;
    this.expectedGames = 0;
    this.moveBudgetMs = 0;
  }

  TournamentRunner.prototype.buildSchedule = function (algoIds) {
    var schedule = [];
    var pairs = tournamentPairs(algoIds);
    var i;
    for (i = 0; i < pairs.length; i += 1) {
      schedule.push([pairs[i][0], pairs[i][1], pairs[i][0]]);
      schedule.push([pairs[i][0], pairs[i][1], pairs[i][1]]);
    }
    return schedule;
  };

  TournamentRunner.prototype.startTournament = function (moveBudgetMs) {
    var algoIds = this.registry.tournamentAlgoIds();
    this.expectedGames = tournamentGameCount(algoIds);
    this.moveBudgetMs = typeof moveBudgetMs === "number" ? moveBudgetMs : 0;
    this.currentResult = createTournamentResult();
    this.pendingGames = this.buildSchedule(algoIds);
  };

  TournamentRunner.prototype.progress = function () {
    if (!this.currentResult) {
      return null;
    }
    return [this.currentResult.gamesPlayed, this.expectedGames];
  };

  TournamentRunner.prototype.pickMove = function (algoId, board, player) {
    var agent = this.registry.getAgent(algoId);
    var budget = this.moveBudgetMs;
    if (algoId === "mcts") {
      return agent.chooseMove(board, player, false, budget);
    }
    if (algoId === "minimax") {
      return agent.chooseMove(board, player, budget);
    }
    if (isNnTimedSearchBot(algoId)) {
      return agent.chooseMoveTimed(board, player, budget);
    }
    if (algoId === "qtable" || algoId === "sarsa" || algoId === "menace") {
      return tabularChooseMove(agent, board, player, false);
    }
    if (algoId === "genetic_menace") {
      return tabularChooseMove(this.registry.genetic_menace.champion(), board, player, false);
    }
    return agent.chooseMove(board, player, false);
  };

  TournamentRunner.prototype.playMatch = function (algoA, algoB, markA) {
    var markB = G.other(markA);
    var board = G.emptyBoard();
    var current = X;
    while (true) {
      var moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      var move;
      if (current === markA) {
        move = this.pickMove(algoA, board, current);
      } else {
        move = this.pickMove(algoB, board, current);
      }
      board = G.applyMove(board, current, move);
      var winner = G.findWinner(board);
      if (winner || G.isDraw(board)) {
        return winner;
      }
      current = G.other(current);
    }
    return G.findWinner(board);
  };

  TournamentRunner.prototype.scoreMatch = function (result, algoA, algoB, markA, winner) {
    var markB = G.other(markA);
    if (!winner) {
      result.points[algoA] += C.SCORE_DRAW;
      result.points[algoB] += C.SCORE_DRAW;
      result.draws[algoA] += 1;
      result.draws[algoB] += 1;
      return;
    }
    if (winner === markA) {
      result.points[algoA] += C.SCORE_WIN;
      result.losses[algoB] += 1;
      result.wins[algoA] += 1;
      result.lossTo[algoB][algoA] = (result.lossTo[algoB][algoA] || 0) + 1;
    } else {
      result.points[algoB] += C.SCORE_WIN;
      result.losses[algoA] += 1;
      result.wins[algoB] += 1;
      result.lossTo[algoA][algoB] = (result.lossTo[algoA][algoB] || 0) + 1;
    }
  };

  TournamentRunner.prototype.playOneGame = function () {
    if (!this.currentResult || !this.pendingGames.length) {
      return false;
    }
    var game = this.pendingGames.shift();
    var winner = this.playMatch(game[0], game[1], game[2]);
    this.scoreMatch(this.currentResult, game[0], game[1], game[2], winner);
    this.currentResult.gamesPlayed += 1;
    return this.pendingGames.length > 0;
  };

  TournamentRunner.prototype.playUntil = function (deadline) {
    while (performance.now() < deadline && this.pendingGames.length) {
      this.playOneGame();
    }
    return this.pendingGames.length > 0;
  };

  TournamentRunner.prototype.finishTournament = function () {
    if (!this.currentResult) {
      return createTournamentResult();
    }
    var result = this.currentResult;
    this.currentResult = null;
    this.pendingGames = [];
    this.expectedGames = 0;
    return result;
  };

  function formatWinnerLossLines(result) {
    var ranked = tournamentResultRanked(result).filter(function (pair) {
      var aid = pair[0];
      return result.wins[aid] + result.draws[aid] + result.losses[aid] > 0;
    });
    if (!ranked.length) {
      return [];
    }
    var topPts = ranked[0][1];
    var winners = ranked.filter(function (pair) { return pair[1] === topPts; }).map(function (p) { return p[0]; });
    var lines = [];
    var w;
    for (w = 0; w < winners.length; w += 1) {
      var winner = winners[w];
      var losses = result.lossTo[winner] || {};
      var keys = Object.keys(losses);
      if (!keys.length) {
        lines.push(displayName(winner) + " (winner): no losses this round");
        continue;
      }
      keys.sort(function (a, b) {
        if (losses[b] !== losses[a]) {
          return losses[b] - losses[a];
        }
        return a < b ? -1 : a > b ? 1 : 0;
      });
      var parts = keys.map(function (opp) {
        return displayName(opp) + " x" + losses[opp];
      });
      lines.push(displayName(winner) + " (winner) lost to: " + parts.join(", "));
    }
    return lines;
  }

  function tournamentParticipants(result) {
    var out = [];
    var i;
    for (i = 0; i < ALL_ALGO_IDS.length; i += 1) {
      var aid = ALL_ALGO_IDS[i];
      if (result.wins[aid] + result.draws[aid] + result.losses[aid] > 0) {
        out.push(aid);
      }
    }
    return out;
  }

  function formatTournamentLines(result) {
    var lines = [];
    var ranked = tournamentResultRanked(result).filter(function (pair) {
      var aid = pair[0];
      return result.wins[aid] + result.draws[aid] + result.losses[aid] > 0;
    });
    var i;
    for (i = 0; i < ranked.length; i += 1) {
      var algoId = ranked[i][0];
      var pts = ranked[i][1];
      lines.push(displayName(algoId) + ": " + pts.toFixed(1) + " pts (" +
        result.wins[algoId] + "-" + result.draws[algoId] + "-" + result.losses[algoId] + ")");
    }
    var extra = formatWinnerLossLines(result);
    for (i = 0; i < extra.length; i += 1) {
      lines.push(extra[i]);
    }
    return lines;
  }

  function formatHistoryEntryLine(entry) {
    var pts = pointsFromHistoryEntry(entry);
    var keys;
    if (entry.participants && entry.participants.length) {
      keys = entry.participants.slice();
    } else {
      keys = Object.keys(pts).filter(function (aid) {
        return pts[aid] > 0;
      });
    }
    keys.sort(function (a, b) { return pts[b] - pts[a]; });
    var parts = [];
    var hi;
    for (hi = 0; hi < keys.length; hi += 1) {
      parts.push(displayName(keys[hi]) + ":" + pts[keys[hi]].toFixed(1));
    }
    var prefix = entry.tournament_number ? "#" + entry.tournament_number + " " : "";
    return prefix + parts.join(" | ");
  }

  function pointsFromHistoryEntry(entry) {
    if (entry.points) {
      var out = {};
      var k;
      for (k in entry.points) {
        if (Object.prototype.hasOwnProperty.call(entry.points, k)) {
          out[k] = entry.points[k];
        }
      }
      return out;
    }
    var pts = {};
    var ranked = entry.ranked || [];
    for (var i = 0; i < ranked.length; i += 1) {
      pts[ranked[i][0]] = ranked[i][1];
    }
    return pts;
  }

  function rollingTournamentTotals(history, windowSize) {
    var totals = {};
    var start = history.length - windowSize;
    if (start < 0) {
      start = 0;
    }
    var i;
    for (i = start; i < history.length; i += 1) {
      var entry = history[i];
      var pts = pointsFromHistoryEntry(entry);
      var ids;
      if (entry.participants && entry.participants.length) {
        ids = entry.participants;
      } else {
        ids = Object.keys(pts);
      }
      var j;
      for (j = 0; j < ids.length; j += 1) {
        var aid = ids[j];
        if (!Object.prototype.hasOwnProperty.call(pts, aid)) {
          continue;
        }
        if (!Object.prototype.hasOwnProperty.call(totals, aid)) {
          totals[aid] = 0;
        }
        totals[aid] += pts[aid];
      }
    }
    return totals;
  }

  function formatRollingTotalsLines(history, windowSize) {
    var totals = rollingTournamentTotals(history, windowSize);
    var ranked = [];
    var k;
    for (k in totals) {
      if (Object.prototype.hasOwnProperty.call(totals, k)) {
        ranked.push([k, totals[k]]);
      }
    }
    ranked.sort(function (a, b) { return b[1] - a[1]; });
    var count = history.length < windowSize ? history.length : windowSize;
    var lines = ["(sum of last " + count + " tournament" + (count === 1 ? "" : "s") + ")"];
    for (var i = 0; i < ranked.length; i += 1) {
      lines.push(displayName(ranked[i][0]) + ": " + ranked[i][1].toFixed(1));
    }
    return lines;
  }

  function canDropMore(registry) {
    return registry.activeTrainingIds().length > C.MIN_ACTIVE_TRAINING_BOTS;
  }

  function tournamentPoints(result, algoId) {
    return result.points[algoId] || 0;
  }

  function protectedActive(registry, result) {
    var active = registry.activeTrainingIds();
    if (active.length <= C.MIN_ACTIVE_TRAINING_BOTS) {
      return active.slice();
    }
    var ranked = active.slice().sort(function (a, b) {
      return tournamentPoints(result, b) - tournamentPoints(result, a);
    });
    var cutoff = tournamentPoints(result, ranked[2]);
    var out = [];
    var i;
    for (i = 0; i < ranked.length; i += 1) {
      if (tournamentPoints(result, ranked[i]) >= cutoff) {
        out.push(ranked[i]);
      }
    }
    var bestNn = null;
    var bestNnPts = -Infinity;
    for (i = 0; i < ranked.length; i += 1) {
      var aid = ranked[i];
      if (!isNnLearner(aid)) {
        continue;
      }
      var pts = tournamentPoints(result, aid);
      if (pts > bestNnPts) {
        bestNnPts = pts;
        bestNn = aid;
      }
    }
    if (bestNn && out.indexOf(bestNn) < 0) {
      out.push(bestNn);
    }
    return out;
  }

  function droppableUnprotected(registry, result) {
    var protectedIds = protectedActive(registry, result);
    var active = registry.activeTrainingIds();
    var out = [];
    var i;
    for (i = 0; i < active.length; i += 1) {
      if (protectedIds.indexOf(active[i]) < 0) {
        out.push(active[i]);
      }
    }
    return out;
  }

  function bottomTier(candidates, result) {
    if (!candidates.length) {
      return [];
    }
    var minPts = tournamentPoints(result, candidates[0]);
    var i;
    for (i = 1; i < candidates.length; i += 1) {
      var pts = tournamentPoints(result, candidates[i]);
      if (pts < minPts) {
        minPts = pts;
      }
    }
    return candidates.filter(function (aid) {
      return tournamentPoints(result, aid) === minPts;
    });
  }

  function tryConsumeDrop(registry, result, tournamentNumber) {
    if (!canDropMore(registry)) {
      return null;
    }
    var droppable = droppableUnprotected(registry, result);
    if (!droppable.length) {
      return null;
    }
    var tier = bottomTier(droppable, result);
    if (tier.length !== 1) {
      return null;
    }
    registry.states[tier[0]].droppedAfterTournament = tournamentNumber;
    return tier[0];
  }

  function earnsDropSlot(tournamentNumber) {
    if (tournamentNumber < C.DROP_FIRST_TOURNAMENT) {
      return false;
    }
    if (tournamentNumber === C.DROP_FIRST_TOURNAMENT) {
      return true;
    }
    return (tournamentNumber - C.DROP_FIRST_TOURNAMENT) % C.DROP_INTERVAL === 0;
  }

  function processElimination(registry, result, tournamentNumber, dropsAvailable) {
    var dropped = [];
    if (!C.ELIMINATION_ENABLED) {
      return { dropsAvailable: dropsAvailable, dropped: dropped };
    }
    if (!canDropMore(registry)) {
      return { dropsAvailable: 0, dropped: dropped };
    }
    if (tournamentNumber > 0 && earnsDropSlot(tournamentNumber)) {
      dropsAvailable += 1;
    }
    if (dropsAvailable > 0) {
      var dropId = tryConsumeDrop(registry, result, tournamentNumber);
      if (dropId !== null) {
        dropped.push(dropId);
        dropsAvailable -= 1;
      }
    }
    return { dropsAvailable: dropsAvailable, dropped: dropped };
  }

  function formatTrainingStatusTable(registry, latestPoints) {
    var rows = [];
    var ordered = trainingStatusOrder(registry, latestPoints);
    var i;
    for (i = 0; i < ordered.length; i += 1) {
      var algoId = ordered[i];
      var st = registry.states[algoId];
      var name = displayName(algoId);
      var greedy;
      if (st.greedyRecent.length) {
        greedy = Math.round(st.greedyWinRate() * 100) + "%";
      } else {
        greedy = st.isActive() ? "pending" : "-";
      }
      var level = st.isActive() ?
        (tabularTrainingDone(registry, algoId) ? "done (no train)" : st.phaseLabel()) :
        "dropped@" + st.droppedAfterTournament;
      rows.push([
        name,
        level,
        String(st.gamesPlayed),
        String(st.nemesis.activeCount()),
        greedy,
        registry.extraStats(algoId)
      ]);
    }
    var headers = ["Algorithm", "Level", "Games", "Nem", "Greedy", "Extra"];
    var widths = headers.map(function (h) { return h.length; });
    for (i = 0; i < rows.length; i += 1) {
      for (var c = 0; c < rows[i].length; c += 1) {
        if (rows[i][c].length > widths[c]) {
          widths[c] = rows[i][c].length;
        }
      }
    }
    function fmtRow(cells) {
      var parts = [];
      for (var j = 0; j < cells.length; j += 1) {
        var cell = cells[j];
        while (cell.length < widths[j]) {
          cell += " ";
        }
        parts.push(cell);
      }
      return parts.join(" | ");
    }
    var divider = widths.map(function (w) {
      var s = "";
      while (s.length < w) {
        s += "-";
      }
      return s;
    }).join("-+-");
    var lines = [fmtRow(headers), divider];
    for (i = 0; i < rows.length; i += 1) {
      lines.push(fmtRow(rows[i]));
    }
    return lines.join("\n");
  }

  function buildCheckpointPayload(
    registry, tournamentHistory, rotationIndex, tournamentNumber, dropsAvailable, latestResult, saveOpts, saveTier, tournamentMoveBudgetMs, trainingSliceScale
  ) {
    return {
      version: 4,
      save_tier: saveTier || "full",
      agents: registry.agentsDictForSave(saveOpts),
      states: registry.allStatesDict(),
      tournament_history: tournamentHistory.slice(-C.TOURNAMENT_HISTORY_LEN),
      rotation_index: rotationIndex,
      tournament_number: tournamentNumber,
      drops_available: dropsAvailable,
      latest_result: latestResult || null,
      tournament_move_budget_ms: typeof tournamentMoveBudgetMs === "number" ?
        tournamentMoveBudgetMs : null,
      training_slice_scale: typeof trainingSliceScale === "number" ? trainingSliceScale : 1
    };
  }

  var SAVE_TIERS = [
    { tabularMaxStates: 6000, label: "tabular6k" },
    { tabularMaxStates: 4000, label: "tabular4k" },
    { tabularMaxStates: 2500, label: "tabular2.5k" },
    { tabularMaxStates: 1500, label: "tabular1.5k" },
    { tabularMaxStates: 1000, label: "tabular1k" }
  ];

  var IDB_NAME = "ConnectFourShapeSearch";
  var IDB_STORE = "checkpoints";
  var IDB_KEY = "learnC4_js_v1";

  function checkpointIdbSupported() {
    return typeof indexedDB !== "undefined";
  }

  function openCheckpointDb(callback) {
    var req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = function (ev) {
      ev.target.result.createObjectStore(IDB_STORE);
    };
    req.onsuccess = function () {
      callback(null, req.result);
    };
    req.onerror = function () {
      callback(req.error || new Error("indexedDB open failed"));
    };
  }

  function idbGetCheckpoint(callback) {
    if (!checkpointIdbSupported()) {
      callback(null, null);
      return;
    }
    openCheckpointDb(function (err, db) {
      if (err) {
        callback(err, null);
        return;
      }
      var tx = db.transaction(IDB_STORE, "readonly");
      var req = tx.objectStore(IDB_STORE).get(IDB_KEY);
      req.onsuccess = function () {
        db.close();
        callback(null, req.result || null);
      };
      req.onerror = function () {
        db.close();
        callback(req.error, null);
      };
    });
  }

  function idbSetCheckpoint(json, callback) {
    openCheckpointDb(function (err, db) {
      if (err) {
        callback(err);
        return;
      }
      var tx = db.transaction(IDB_STORE, "readwrite");
      tx.objectStore(IDB_STORE).put(json, IDB_KEY);
      tx.oncomplete = function () {
        db.close();
        callback(null);
      };
      tx.onerror = function () {
        db.close();
        callback(tx.error);
      };
    });
  }

  function idbDeleteCheckpoint(callback) {
    if (!checkpointIdbSupported()) {
      if (callback) {
        callback(null);
      }
      return;
    }
    openCheckpointDb(function (err, db) {
      if (err) {
        if (callback) {
          callback(err);
        }
        return;
      }
      var tx = db.transaction(IDB_STORE, "readwrite");
      tx.objectStore(IDB_STORE).delete(IDB_KEY);
      tx.oncomplete = function () {
        db.close();
        if (callback) {
          callback(null);
        }
      };
      tx.onerror = function () {
        db.close();
        if (callback) {
          callback(tx.error);
        }
      };
    });
  }

  function writeStoragePointer(meta) {
    try {
      localStorage.setItem(C.STORAGE_KEY, JSON.stringify(meta));
    } catch (e) {
      /* optional when IndexedDB holds the checkpoint */
    }
  }

  function readStoredCheckpointRaw(callback) {
    var inline = null;
    try {
      var raw = localStorage.getItem(C.STORAGE_KEY);
      if (raw) {
        var parsed = JSON.parse(raw);
        if (parsed && parsed.backend === "idb") {
          if (callback) {
            idbGetCheckpoint(function (err, json) {
              if (err) {
                callback(err, null);
                return;
              }
              callback(null, json);
            });
          }
          return null;
        }
        if (parsed && parsed.agents) {
          inline = raw;
        }
      }
    } catch (e) {
      /* fall through */
    }
    if (inline !== null) {
      if (callback) {
        callback(null, inline);
      }
      return inline;
    }
    if (callback && checkpointIdbSupported()) {
      idbGetCheckpoint(function (err, json) {
        callback(err, json);
      });
      return null;
    }
    if (callback) {
      callback(null, null);
    }
    return null;
  }

  function applyCheckpointData(registry, data) {
    registry.loadAgentsDict(data.agents);
    registry.loadStatesDict(data.states);
    var history = data.tournament_history || [];
    return {
      history: history,
      rotation: data.rotation_index || 0,
      tournamentNumber: typeof data.tournament_number === "number" ?
        data.tournament_number : history.length,
      dropsAvailable: typeof data.drops_available === "number" ?
        data.drops_available : (data.pending_drops ? data.pending_drops.length : 0),
      latestResult: hydrateTournamentResult(data.latest_result),
      tournamentMoveBudgetMs: typeof data.tournament_move_budget_ms === "number" ?
        data.tournament_move_budget_ms : null,
      trainingSliceScale: typeof data.training_slice_scale === "number" ?
        data.training_slice_scale : 1,
      loadError: null
    };
  }

  function buildCheckpointJson(
    registry, tournamentHistory, rotationIndex, tournamentNumber, dropsAvailable, latestResult, tiers, tournamentMoveBudgetMs, trainingSliceScale
  ) {
    var i;
    for (i = 0; i < tiers.length; i += 1) {
      var tier = tiers[i];
      var saveOpts = { tabularMaxStates: tier.tabularMaxStates };
      var payload = buildCheckpointPayload(
        registry, tournamentHistory, rotationIndex, tournamentNumber, dropsAvailable, latestResult,
        saveOpts, tier.label, tournamentMoveBudgetMs, trainingSliceScale
      );
      var json = JSON.stringify(payload);
      if (json.length <= C.SAVE_SIZE_LIMIT || i === tiers.length - 1) {
        return {
          json: json,
          bytes: json.length,
          trimmed: i > 0,
          tier: tier.label
        };
      }
    }
    return null;
  }

  function saveCheckpointToLocalStorage(
    registry, tournamentHistory, rotationIndex, tournamentNumber, dropsAvailable, latestResult, tournamentMoveBudgetMs, trainingSliceScale
  ) {
    var built = buildCheckpointJson(
      registry, tournamentHistory, rotationIndex, tournamentNumber, dropsAvailable, latestResult, SAVE_TIERS,
      tournamentMoveBudgetMs, trainingSliceScale
    );
    if (!built) {
      return {
        ok: false,
        error: "save failed",
        bytes: 0,
        trimmed: false,
        tier: null,
        backend: "localStorage"
      };
    }
    try {
      localStorage.setItem(C.STORAGE_KEY, built.json);
      return {
        ok: true,
        bytes: built.bytes,
        trimmed: built.trimmed,
        tier: built.tier,
        backend: "localStorage"
      };
    } catch (e) {
      return {
        ok: false,
        error: String(e.message || e),
        bytes: built.bytes,
        trimmed: built.trimmed,
        tier: built.tier,
        backend: "localStorage"
      };
    }
  }

  function saveCheckpoint(
    registry, tournamentHistory, rotationIndex, tournamentNumber, dropsAvailable, latestResult, tournamentMoveBudgetMs, trainingSliceScale, done
  ) {
    if (checkpointIdbSupported()) {
      var fullTier = [{ tabularMaxStates: null, label: "full" }];
      var built = buildCheckpointJson(
        registry, tournamentHistory, rotationIndex, tournamentNumber, dropsAvailable, latestResult, fullTier,
        tournamentMoveBudgetMs, trainingSliceScale
      );
      if (!built) {
        built = buildCheckpointJson(
          registry, tournamentHistory, rotationIndex, tournamentNumber, dropsAvailable, latestResult, SAVE_TIERS,
          tournamentMoveBudgetMs, trainingSliceScale
        );
      }
      if (!built) {
        var missing = {
          ok: false,
          error: "save failed",
          bytes: 0,
          trimmed: false,
          tier: null,
          backend: "indexedDB"
        };
        if (done) {
          done(missing);
        }
        return missing;
      }
      idbSetCheckpoint(built.json, function (err) {
        var status;
        if (err) {
          status = {
            ok: false,
            error: String(err.message || err),
            bytes: built.bytes,
            trimmed: built.trimmed,
            tier: built.tier,
            backend: "indexedDB"
          };
        } else {
          writeStoragePointer({
            backend: "idb",
            version: 4,
            bytes: built.bytes,
            tier: built.tier,
            savedAt: Date.now()
          });
          status = {
            ok: true,
            bytes: built.bytes,
            trimmed: built.trimmed,
            tier: built.tier,
            backend: "indexedDB"
          };
        }
        if (done) {
          done(status);
        }
      });
      return { ok: true, pending: true, bytes: built.bytes, backend: "indexedDB" };
    }
    var syncResult = saveCheckpointToLocalStorage(
      registry, tournamentHistory, rotationIndex, tournamentNumber, dropsAvailable, latestResult,
      tournamentMoveBudgetMs, trainingSliceScale
    );
    if (done) {
      done(syncResult);
    }
    return syncResult;
  }

  function emptyLoadedMeta() {
    return {
      history: [],
      rotation: 0,
      tournamentNumber: 0,
      dropsAvailable: 0,
      latestResult: null,
      tournamentMoveBudgetMs: null,
      trainingSliceScale: 1,
      loadError: null
    };
  }

  function loadCheckpointSync(registry) {
    var raw = null;
    try {
      raw = localStorage.getItem(C.STORAGE_KEY);
    } catch (e) {
      var blocked = emptyLoadedMeta();
      blocked.loadError = "localStorage unavailable: " + String(e.message || e);
      return blocked;
    }
    if (!raw) {
      return emptyLoadedMeta();
    }
    try {
      var data = JSON.parse(raw);
      if (data && data.backend === "idb") {
        return emptyLoadedMeta();
      }
      try {
        return applyCheckpointData(registry, data);
      } catch (loadErr) {
        return {
          history: [],
          rotation: 0,
          tournamentNumber: 0,
          dropsAvailable: 0,
          latestResult: null,
          tournamentMoveBudgetMs: null,
          loadError: "Could not load saved agents: " + String(loadErr.message || loadErr)
        };
      }
    } catch (err) {
      deleteCheckpointSync();
      var corrupt = emptyLoadedMeta();
      corrupt.loadError = "Corrupt save removed: " + String(err.message || err);
      return corrupt;
    }
  }

  function loadCheckpointAsync(registry, callback) {
    readStoredCheckpointRaw(function (err, raw) {
      if (err) {
        var fail = emptyLoadedMeta();
        fail.loadError = "Could not load checkpoint: " + String(err.message || err);
        callback(fail);
        return;
      }
      if (!raw) {
        callback(emptyLoadedMeta());
        return;
      }
      try {
        callback(applyCheckpointData(registry, JSON.parse(raw)));
      } catch (loadErr) {
        var bad = emptyLoadedMeta();
        bad.loadError = "Could not load saved agents: " + String(loadErr.message || loadErr);
        callback(bad);
      }
    });
  }

  function deleteCheckpointSync() {
    try {
      localStorage.removeItem(C.STORAGE_KEY);
    } catch (e) {
      /* ignore */
    }
  }

  function deleteCheckpoint(done) {
    deleteCheckpointSync();
    idbDeleteCheckpoint(done || null);
  }

  /* --- Application (legacy multi-bot Training loop; kept for benches/tests) --- */
  function Application(deferLoad) {
    this.registry = new AgentRegistry();
    this.training = new TrainingEngine(this.registry);
    this.tournament = new TournamentRunner(this.registry);
    this.tournamentHistory = [];
    this.latestResult = null;
    this.tournamentNumber = 0;
    this.dropsAvailable = 0;
    this.tournamentMoveBudgetMs = null;
    this.tournamentStartedAt = null;
    this.lastTournamentDurationMs = null;
    this.lastSaveAt = performance.now();
    this.lastSaveStatus = null;
    this.loadWarning = null;
    this.lastRuntimeError = null;
    this.phase = "training";
    this.running = false;
    this.trainingPaused = false;
    this.rafId = null;

    if (!deferLoad) {
      this.applyLoaded(loadCheckpointSync(this.registry));
    }
  }

  Application.prototype.captureRuntimeError = function (err, where) {
    var report = formatErrorReport(err, this);
    this.lastRuntimeError = {
      message: err && err.message ? err.message : String(err),
      where: where || "unknown",
      report: report,
      at: Date.now()
    };
    if (typeof console !== "undefined" && console.error) {
      console.error("[C4 " + this.lastRuntimeError.where + "]", report);
    }
    return this.lastRuntimeError;
  };

  Application.prototype.applyLoaded = function (loaded) {
    this.tournamentHistory = loaded.history;
    this.training.rotationIndex = loaded.rotation;
    this.tournamentNumber = loaded.tournamentNumber;
    this.dropsAvailable = loaded.dropsAvailable;
    this.tournamentMoveBudgetMs = loaded.tournamentMoveBudgetMs;
    this.latestResult = loaded.latestResult;
    this.loadWarning = loaded.loadError;
    if (loaded.history.length) {
      var lastPoints = pointsFromHistoryEntry(loaded.history[loaded.history.length - 1]);
      this.training.updateTournamentRankings(lastPoints);
    }
    purgeIneligibleNemesis(this.registry);
    this.training.sliceScale = 1;
    this.training.beginTrainingRound();
  };

  Application.prototype.activityLabel = function () {
    if (this.phase === "tournament") {
      var progress = this.tournament.progress();
      var budget = this.tournament.moveBudgetMs;
      var budgetLabel = typeof budget === "number" && budget > 0 ?
        " " + budget.toFixed(2) + "ms/move" : "";
      if (progress) {
        return "TOURNAMENT " + progress[0] + "/" + progress[1] + budgetLabel;
      }
      return "TOURNAMENT" + budgetLabel;
    }
    if (this.phase === "greedy_eval") {
      var greedyProgress = this.training.greedyEvalProgress();
      if (greedyProgress) {
        return "GREEDY EVAL " + greedyProgress;
      }
      return "GREEDY EVAL";
    }
    return "Training: " + displayName(this.training.currentAlgoId());
  };

  Application.prototype.saveNow = function (done, reason) {
    var self = this;
    var result = saveCheckpoint(
      this.registry,
      this.tournamentHistory,
      this.training.rotationIndex,
      this.tournamentNumber,
      this.dropsAvailable,
      this.latestResult,
      this.tournamentMoveBudgetMs,
      this.training.sliceScale,
      function (status) {
        if (reason) {
          status.reason = reason;
        }
        self.lastSaveStatus = status;
        if (status.ok) {
          self.lastSaveAt = performance.now();
        }
        if (done) {
          done(status);
        }
      }
    );
    if (!result.pending) {
      if (reason) {
        result.reason = reason;
      }
      this.lastSaveStatus = result;
      if (result.ok) {
        this.lastSaveAt = performance.now();
      }
      if (done) {
        done(result);
      }
    } else if (reason) {
      result.reason = reason;
      this.lastSaveStatus = result;
    }
    return this.lastSaveStatus;
  };

  Application.prototype.maybeSave = function () {
    if ((performance.now() - this.lastSaveAt) / 1000 >= C.AUTO_SAVE_SECONDS) {
      this.saveNow(null, "timer");
    }
  };

  Application.prototype.resetAll = function () {
    deleteCheckpoint();
    this.registry = new AgentRegistry();
    this.training = new TrainingEngine(this.registry);
    this.tournament = new TournamentRunner(this.registry);
    this.tournamentHistory = [];
    this.latestResult = null;
    this.tournamentNumber = 0;
    this.dropsAvailable = 0;
    this.tournamentMoveBudgetMs = null;
    this.tournamentStartedAt = null;
    this.lastTournamentDurationMs = null;
    this.phase = "training";
    this.saveNow();
  };

  Application.prototype.startTournamentRound = function () {
    this.phase = "tournament";
    var expectedGames = tournamentGameCount(this.registry.tournamentAlgoIds());
    if (this.tournamentMoveBudgetMs === null || this.tournamentMoveBudgetMs <= 0) {
      this.tournamentMoveBudgetMs = computeInitialTournamentMoveBudgetMs(expectedGames);
    }
    this.tournament.startTournament(this.tournamentMoveBudgetMs);
    this.tournamentStartedAt = performance.now();
  };

  Application.prototype.finishTournamentRound = function () {
    this.latestResult = this.tournament.finishTournament();
    var actualMs = this.tournamentStartedAt ?
      performance.now() - this.tournamentStartedAt : 0;
    this.lastTournamentDurationMs = actualMs;
    if (actualMs > 0 && this.tournamentMoveBudgetMs > 0) {
      this.tournamentMoveBudgetMs = adjustTournamentMoveBudget(this.tournamentMoveBudgetMs, actualMs);
    }
    this.tournamentStartedAt = null;
    this.tournamentNumber += 1;
    var pointsCopy = {};
    var participants = tournamentParticipants(this.latestResult);
    var i;
    for (i = 0; i < participants.length; i += 1) {
      var pid = participants[i];
      pointsCopy[pid] = this.latestResult.points[pid];
    }
    this.tournamentHistory.push({
      tournament_number: this.tournamentNumber,
      ranked: this.latestResult.ranked().filter(function (pair) {
        return participants.indexOf(pair[0]) >= 0;
      }),
      points: pointsCopy,
      participants: participants,
      ts: Date.now() / 1000
    });
    var elim = processElimination(
      this.registry, this.latestResult, this.tournamentNumber, this.dropsAvailable
    );
    this.dropsAvailable = elim.dropsAvailable;
    this.training.updateTournamentRankings(this.latestResult.points);
    this.training.ensureCurrentActive();
    this.phase = "greedy_eval";
    this.training.startGreedyEvalRound();
    this.saveNow(null, "tournament");
  };

  Application.prototype.finishGreedyEvalRound = function () {
    this.training.resetRoundTrainingTimer();
    this.phase = "training";
    this.saveNow(null, "tournament");
  };

  Application.prototype.setTrainingPaused = function (paused) {
    this.trainingPaused = !!paused;
  };

  Application.prototype.pickPlayMove = function (algoId, board, player, budgetSec) {
    return pickPlayMove(this.registry, algoId, board, player, budgetSec);
  };

  Application.prototype.runWorkUntil = function (deadline) {
    while (performance.now() < deadline) {
      tracePush("runWorkUntil", { phase: this.phase });
      try {
        if (this.phase === "training") {
          if (!this.training.runTrainingUntil(deadline)) {
            this.startTournamentRound();
          }
        } else if (this.phase === "tournament") {
          if (!this.tournament.playUntil(deadline)) {
            this.finishTournamentRound();
          }
        } else if (this.phase === "greedy_eval") {
          if (!this.training.runGreedyUntil(deadline)) {
            this.finishGreedyEvalRound();
          }
        }
      } finally {
        tracePop();
      }
    }
  };

  Application.prototype.getSnapshot = function () {
    var activity = this.activityLabel();
    var sliceLimit = this.training.roundSliceLimit;
    var latestTournamentHeading = this.tournamentNumber > 0 ?
      "Latest tournament #" + this.tournamentNumber :
      "Latest tournament";
    var lastSaveAgo = (performance.now() - this.lastSaveAt) / 1000;

    var tournTrainLabel;
    if (this.training.shouldStartTournament()) {
      tournTrainLabel = "tourn next";
    } else {
      tournTrainLabel = this.training.estimatedRotationRemaining().toFixed(1) + "s";
    }

    var statusLine = activity + " | slice " +
      this.training.sliceElapsed().toFixed(1) + "/" + sliceLimit.toFixed(1) +
      "s train | next tourn " +
      tournTrainLabel + " | last save " +
      Math.round(lastSaveAgo) + "s ago";
    if (typeof this.tournamentMoveBudgetMs === "number" && this.tournamentMoveBudgetMs > 0) {
      statusLine += " | tourn " + this.tournamentMoveBudgetMs.toFixed(2) + "ms/move";
      if (typeof this.lastTournamentDurationMs === "number" && this.lastTournamentDurationMs > 0) {
        statusLine += " (last " + (this.lastTournamentDurationMs / 1000).toFixed(1) + "s)";
      }
    }

    var tournamentText;
    if (this.phase === "tournament" && this.tournament.progress()) {
      var tp = this.tournament.progress();
      tournamentText = "Tournament in progress: " + tp[0] + "/" + tp[1] + " games...";
    } else if (this.latestResult) {
      tournamentText = formatTournamentLines(this.latestResult).join("\n");
    } else if (!this.tournamentHistory.length) {
      tournamentText = "Waiting for first tournament...";
    } else {
      tournamentText = "Waiting for first tournament...";
    }

    var rollingText = this.tournamentHistory.length ?
      formatRollingTotalsLines(this.tournamentHistory, C.ROLLING_TOURNAMENT_WINDOW).join("\n") :
      "(no tournaments yet)";

    var trainingTable = formatTrainingStatusTable(
      this.registry, this.training.latestTournamentPoints
    );

    var histStart = this.tournamentHistory.length - C.TOURNAMENT_HISTORY_LEN;
    if (histStart < 0) {
      histStart = 0;
    }
    var histLines = [];
    for (var h = histStart; h < this.tournamentHistory.length; h += 1) {
      histLines.push(formatHistoryEntryLine(this.tournamentHistory[h]));
    }
    var historyText = histLines.length ? histLines.join("\n") : "(none yet)";

    var footerText = "IndexedDB " + C.STORAGE_KEY +
      " | Auto-save after each tournament (" + C.AUTO_SAVE_SECONDS + "s fallback).";
    if (this.lastSaveStatus) {
      if (this.lastSaveStatus.pending) {
        footerText += " | Saving...";
      } else if (this.lastSaveStatus.ok) {
        footerText += " | Last save OK (" + Math.round(this.lastSaveStatus.bytes / 1024) + " KB";
        if (this.lastSaveStatus.backend) {
          footerText += ", " + this.lastSaveStatus.backend;
        }
        if (this.lastSaveStatus.trimmed) {
          footerText += ", " + this.lastSaveStatus.tier;
        }
        if (this.lastSaveStatus.reason === "tournament") {
          footerText += ", after tournament";
        } else if (this.lastSaveStatus.reason === "timer") {
          footerText += ", timer";
        }
        footerText += ").";
      } else {
        footerText += " | Save FAILED: " + this.lastSaveStatus.error;
      }
    }

    return {
      statusLine: statusLine,
      latestTournamentHeading: latestTournamentHeading,
      tournamentText: tournamentText,
      rollingText: rollingText,
      trainingTable: trainingTable,
      historyText: historyText,
      footerText: footerText,
      loadWarning: this.loadWarning,
      saveStatus: this.lastSaveStatus,
      runtimeError: this.lastRuntimeError
    };
  };

  Application.prototype.stop = function () {
    this.running = false;
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.saveNow();
  };

  function init(options) {
    options = options || {};
    var app = new Application(true);
    var lastUi = performance.now();
    var refreshMs = C.UI_REFRESH_SECONDS * 1000;
    var workBudgetMs = 50;
    var loopStarted = false;

    function publishSnapshot() {
      if (options.onSnapshot) {
        options.onSnapshot(app.getSnapshot());
      }
    }

    function startLoop() {
      if (loopStarted) {
        return;
      }
      loopStarted = true;
      function loop() {
        if (!app.running) {
          return;
        }
        try {
          var now = performance.now();
          if (now - lastUi >= refreshMs) {
            publishSnapshot();
            lastUi = now;
            if (!app.trainingPaused) {
              app.maybeSave();
            }
          }
          if (!app.trainingPaused) {
            app.runWorkUntil(now + workBudgetMs);
          }
        } catch (loopErr) {
          var captured = app.captureRuntimeError(loopErr, "mainLoop");
          if (options.onError) {
            options.onError(loopErr, captured.report);
          } else {
            throw loopErr;
          }
        }
        if (typeof requestAnimationFrame === "function") {
          app.rafId = requestAnimationFrame(loop);
        } else {
          app.rafId = setTimeout(loop, refreshMs);
        }
      }
      if (typeof requestAnimationFrame === "function") {
        app.rafId = requestAnimationFrame(loop);
      } else {
        app.rafId = setTimeout(loop, 0);
      }
    }

    function boot(loaded) {
      try {
        app.applyLoaded(loaded);
        if (options.reset) {
          app.resetAll();
        }
        app.running = true;
        publishSnapshot();
        app.saveNow(function () {
          publishSnapshot();
          startLoop();
        }, "startup");
      } catch (bootErr) {
        var captured = app.captureRuntimeError(bootErr, "boot");
        if (options.onError) {
          options.onError(bootErr, captured.report);
        }
      }
    }

    publishSnapshot();
    if (checkpointIdbSupported()) {
      loadCheckpointAsync(app.registry, boot);
    } else {
      boot(loadCheckpointSync(app.registry));
    }

    if (typeof document !== "undefined") {
      window.addEventListener("pagehide", function () {
        app.saveNow();
      });
      window.addEventListener("beforeunload", function () {
        app.saveNow();
      });
    }

    return app;
  }

  /**
   * Shape-lab helpers: create agents by family, train/eval without fixed TRAINING_ALGO_IDS.
   */
  var SHAPE_FAMILIES = {
    nn_policy: { kind: "policy", builder: function (sizes, lr) { return new NnAgent(sizes, lr); } },
    nn_value: { kind: "value", builder: function (sizes, lr) { return new NnValueAgent(sizes, lr); } },
    reinforce_policy: { kind: "policy", builder: function (sizes, lr) { return new ReinforceAgent(lr, sizes); } },
    reinforce_value: { kind: "value", builder: function (sizes, lr) { return new ReinforceValueAgent(sizes, lr); } },
    tfjs_policy: { kind: "policy", builder: function (sizes, lr) {
      var a = new TfjsPolicyAgent(sizes);
      if (typeof lr === "number") { a.learningRate = lr; if (a.optimizer && a.rebuildOptimizer) { a.rebuildOptimizer(); } }
      return a;
    } },
    tfjs_value: { kind: "value", builder: function (sizes, lr) {
      var a = new TfjsValueAgent(sizes);
      if (typeof lr === "number") { a.learningRate = lr; }
      return a;
    } },
    qtable: { kind: "tabular", builder: function () { return new OPP.QBrain(); } },
    sarsa: { kind: "tabular", builder: function () { return new SarsaBrain(); } },
    menace: { kind: "tabular", builder: function () { return new OPP.MenaceBrain(); } },
    neat: { kind: "neat", builder: function () { return new SimpleGaNeatAgent(); } },
    neat_value: { kind: "neat_value", builder: function () { return new SimpleGaNeatValueAgent(); } },
    genetic_menace: { kind: "genetic", builder: function () { return new GeneticMenaceAgent(); } }
  };

  function defaultLayerSizes(family) {
    var inputP = C.POLICY_INPUT_DIM || 78;
    var inputV = C.VALUE_INPUT_DIM || 77;
    if (family === "nn_policy" || family === "reinforce_policy" || family === "tfjs_policy") {
      return [inputP, 64, inputP, 7];
    }
    if (family === "nn_value" || family === "reinforce_value" || family === "tfjs_value") {
      return [inputV, 64, inputV, 1];
    }
    return null;
  }

  function defaultLearningRate(family) {
    if (family === "tfjs_policy" || family === "tfjs_value") {
      return 0.001;
    }
    if (family === "reinforce_policy" || family === "reinforce_value") {
      return 0.01;
    }
    if (family === "nn_policy" || family === "nn_value") {
      return 0.03;
    }
    return null;
  }

  function applyShapeLearningRate(agent, family, learningRate) {
    if (!agent || typeof learningRate !== "number") {
      return agent;
    }
    agent.learningRate = learningRate;
    if (agent.net) {
      agent.net.learningRate = learningRate;
    }
    if ((family === "tfjs_policy" || family === "tfjs_value") && typeof agent.optimizer !== "undefined" && global.tf && global.tf.train) {
      agent.optimizer = global.tf.train.adam(learningRate);
    }
    return agent;
  }

  function createShapeAgent(family, layerSizes, learningRate) {
    var meta = SHAPE_FAMILIES[family];
    if (!meta) {
      throw new Error("Unknown shape family: " + family);
    }
    var sizes = layerSizes || defaultLayerSizes(family);
    var lr = typeof learningRate === "number" ? learningRate : defaultLearningRate(family);
    var agent = meta.builder(sizes, lr);
    return applyShapeLearningRate(agent, family, lr);
  }

  /**
   * Bot ply cap from FractionalMinimax round level + UI offset.
   * offset 0 = Match Minimax (floor(level)); -1..-10 = shallower. Always >= 1.
   */
  function botSearchDepthForLevel(mmLevel, offset) {
    var base = Math.floor(M.normalizeLevel(mmLevel));
    if (base < 1) {
      base = 1;
    }
    var off = typeof offset === "number" && isFinite(offset) ? Math.floor(offset) : 0;
    if (off > 0) {
      off = 0;
    }
    if (off < -10) {
      off = -10;
    }
    return Math.max(1, base + off);
  }

  function shapeAgentChoose(agent, family, board, player, explore, budgetMs, maxDepth) {
    var meta = SHAPE_FAMILIES[family];
    var depth = typeof maxDepth === "number" && maxDepth > 0 ? maxDepth : null;
    if (meta.kind === "tabular") {
      return tabularChooseMove(agent, board, player, !!explore);
    }
    if (meta.kind === "genetic") {
      return agent.chooseMove(board, player, !!explore);
    }
    if (meta.kind === "neat" || meta.kind === "neat_value") {
      if (depth !== null && agent.chooseMoveTimed) {
        var neatBudget = typeof budgetMs === "number" && budgetMs > 0 ? budgetMs : 1e12;
        return agent.chooseMoveTimed(board, player, neatBudget, depth);
      }
      if (typeof budgetMs === "number" && budgetMs > 0 && agent.chooseMoveTimed) {
        return agent.chooseMoveTimed(board, player, budgetMs);
      }
      return agent.chooseMove(board, player, !!explore, 0);
    }
    if (depth !== null && agent.chooseMoveTimed) {
      var depthBudget = typeof budgetMs === "number" && budgetMs > 0 ? budgetMs : 1e12;
      return agent.chooseMoveTimed(board, player, depthBudget, depth);
    }
    if (typeof budgetMs === "number" && budgetMs > 0 && agent.chooseMoveTimed) {
      return agent.chooseMoveTimed(board, player, budgetMs);
    }
    return agent.chooseMove(board, player, !!explore, 0);
  }

  function shapeBuildStep(agent, family, board, player, move) {
    var meta = SHAPE_FAMILIES[family];
    if (meta.kind === "tabular" || meta.kind === "genetic") {
      return buildTabularStep(board, player, move);
    }
    return agent.buildStep(board, player, move);
  }

  function shapeLearn(agent, family, trajectory, mark, winner) {
    var meta = SHAPE_FAMILIES[family];
    if (!trajectory.length) {
      return;
    }
    if (meta.kind === "tabular") {
      if (family === "menace") {
        agent.learnFromTrajectory(trajectory, winner);
      } else {
        agent.learnFromTrajectory(trajectory, mark, winner);
      }
      return;
    }
    if (meta.kind === "genetic") {
      return;
    }
    if (meta.kind === "neat" || meta.kind === "neat_value") {
      return;
    }
    agent.learnFromTrajectory(trajectory, mark, winner);
  }

  function shapeTrainVsRandom(agent, family, explore, train, maxDepth) {
    var rng = Math.random.bind(Math);
    var learnerMark = rng() < 0.5 ? X : O;
    var board = G.emptyBoard();
    var current = X;
    var trajectory = [];
    var depth = typeof maxDepth === "number" && maxDepth > 0 ? maxDepth : null;
    while (true) {
      var moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      var move;
      if (current === learnerMark) {
        move = shapeAgentChoose(agent, family, board, current, explore, 0, depth);
        if (train) {
          trajectory.push(shapeBuildStep(agent, family, board, current, move));
        }
      } else {
        move = moves[Math.floor(rng() * moves.length)];
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    var winner = G.findWinner(board);
    if (train) {
      shapeLearn(agent, family, trajectory, learnerMark, winner);
    }
    return G.resultForPlayer(winner, learnerMark);
  }

  /** Self-play: same agent both sides; both trajectories learn. */
  function shapeTrainVsSelf(agent, family, explore, maxDepth) {
    var board = G.emptyBoard();
    var current = X;
    var trajX = [];
    var trajO = [];
    var depth = typeof maxDepth === "number" && maxDepth > 0 ? maxDepth : null;
    while (true) {
      var moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      var move = shapeAgentChoose(agent, family, board, current, explore, 0, depth);
      if (current === X) {
        trajX.push(shapeBuildStep(agent, family, board, current, move));
      } else {
        trajO.push(shapeBuildStep(agent, family, board, current, move));
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    var winner = G.findWinner(board);
    shapeLearn(agent, family, trajX, X, winner);
    shapeLearn(agent, family, trajO, O, winner);
    return winner;
  }

  /**
   * Train vs FractionalMinimax. Optional seed for nemesis replay.
   * Returns { result, seed }.
   */
  function shapeTrainVsMinimax(agent, family, level, explore, train, maxDepth, seed) {
    var usedSeed = typeof seed === "number" ? seed : Math.floor(Math.random() * 2147483647);
    var rng = M.makeSeededRng(usedSeed);
    var opponent = new M.FractionalMinimaxOpponent(M.normalizeLevel(level), rng);
    var learnerMark = rng() < 0.5 ? X : O;
    var board = G.emptyBoard();
    var current = X;
    var trajectory = [];
    var depth = typeof maxDepth === "number" && maxDepth > 0 ? maxDepth : null;
    while (true) {
      var moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      var move;
      if (current === learnerMark) {
        move = shapeAgentChoose(agent, family, board, current, !!explore, 0, depth);
        if (train) {
          trajectory.push(shapeBuildStep(agent, family, board, current, move));
        }
      } else {
        move = opponent.chooseMove(board, current);
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    var winner = G.findWinner(board);
    if (train) {
      shapeLearn(agent, family, trajectory, learnerMark, winner);
    }
    return {
      result: G.resultForPlayer(winner, learnerMark),
      seed: usedSeed
    };
  }

  function shapeEvolveOne(agent, family) {
    if (family === "neat") {
      agent.evolveOneGeneration(function () { return Math.random(); });
      return;
    }
    if (family === "neat_value") {
      agent.evolveOneGeneration(function () { return Math.random(); });
      return;
    }
    if (family === "genetic_menace") {
      var scores = [];
      var games = gaFitnessGames();
      var i;
      var g;
      for (i = 0; i < agent.population.length; i += 1) {
        var wins = 0;
        for (g = 0; g < games; g += 1) {
          wins += fitnessPoints(playMenaceBrainVsRandom(agent.population[i], true, true));
        }
        scores.push(wins / games);
      }
      agent.evolve(scores);
    }
  }

  /**
   * Eval vs FractionalMinimax.
   * If maxDepth is set, learner searches to that ply (wisdom vs MM depth).
   * Otherwise: time each MM move and give learner the same budget.
   * opts.extendTo + opts.extendIfWr: after `evalGames`, if points WR > extendIfWr, keep
   * playing until extendTo games (same cumulative WDL). WR counts draws as 0.5.
   */
  function shapeEvalVsMinimaxTimed(agent, family, level, evalGames, maxDepth, opts) {
    opts = opts || {};
    var wins = 0;
    var draws = 0;
    var losses = 0;
    var normLevel = M.normalizeLevel(level);
    var useDepth = typeof maxDepth === "number" && maxDepth > 0;
    var extendTo = typeof opts.extendTo === "number" && opts.extendTo > evalGames ?
      opts.extendTo : null;
    var extendIfWr = typeof opts.extendIfWr === "number" ? opts.extendIfWr : null;
    var target = Math.max(1, evalGames | 0);
    var g = 0;
    while (g < target) {
      var rng = Math.random.bind(Math);
      var opponent = new M.FractionalMinimaxOpponent(normLevel, rng);
      var learnerMark = rng() < 0.5 ? X : O;
      var board = G.emptyBoard();
      var current = X;
      var lastMmMs = 2;
      while (true) {
        var moves = G.legalMoves(board);
        if (!moves.length) {
          break;
        }
        var move;
        if (current === learnerMark) {
          if (useDepth) {
            move = shapeAgentChoose(agent, family, board, current, false, 0, maxDepth);
          } else {
            move = shapeAgentChoose(agent, family, board, current, false, lastMmMs);
          }
        } else {
          var t0 = performance.now();
          move = opponent.chooseMove(board, current);
          lastMmMs = Math.max(0.5, performance.now() - t0);
        }
        board = G.applyMove(board, current, move);
        if (G.findWinner(board) || G.isDraw(board)) {
          break;
        }
        current = G.other(current);
      }
      var result = G.resultForPlayer(G.findWinner(board), learnerMark);
      if (result === "win") {
        wins += 1;
      } else if (result === "draw") {
        draws += 1;
      } else {
        losses += 1;
      }
      g += 1;
      if (
        extendTo !== null &&
        extendIfWr !== null &&
        g === evalGames &&
        (wins + 0.5 * draws) / g > extendIfWr &&
        target < extendTo
      ) {
        target = extendTo;
      }
    }
    return {
      wins: wins,
      draws: draws,
      losses: losses,
      games: g,
      winRate: (wins + 0.5 * draws) / Math.max(1, g),
      extended: g > evalGames
    };
  }

  /** Two agents play. learnB false = frozen opponent (only A learns). */
  function shapeTrainVsPeer(agentA, familyA, agentB, familyB, explore, maxDepth, learnB) {
    if (typeof learnB !== "boolean") {
      learnB = true;
    }
    var aIsX = Math.random() < 0.5;
    var board = G.emptyBoard();
    var current = X;
    var trajA = [];
    var trajB = [];
    var depth = typeof maxDepth === "number" && maxDepth > 0 ? maxDepth : null;
    while (true) {
      var moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      var aToMove = (current === X) === aIsX;
      var move;
      if (aToMove) {
        move = shapeAgentChoose(agentA, familyA, board, current, explore, 0, depth);
        trajA.push(shapeBuildStep(agentA, familyA, board, current, move));
      } else {
        move = shapeAgentChoose(agentB, familyB, board, current, explore, 0, depth);
        if (learnB) {
          trajB.push(shapeBuildStep(agentB, familyB, board, current, move));
        }
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    var winner = G.findWinner(board);
    var markA = aIsX ? X : O;
    var markB = aIsX ? O : X;
    shapeLearn(agentA, familyA, trajA, markA, winner);
    if (learnB) {
      shapeLearn(agentB, familyB, trajB, markB, winner);
    }
    return winner;
  }

  function shapeEvalPoints(ev) {
    if (!ev) {
      return 0;
    }
    return (ev.wins || 0) + 0.5 * (ev.draws || 0);
  }

  /** One game, explore off, optional ply cap. aIsX: agentA sits as X. No learning. */
  function shapePlayAgents(agentA, familyA, agentB, familyB, aIsX, maxDepth) {
    var board = G.emptyBoard();
    var current = X;
    var depth = typeof maxDepth === "number" && maxDepth > 0 ? maxDepth : 1;
    while (true) {
      var moves = G.legalMoves(board);
      if (!moves.length) {
        break;
      }
      var aToMove = (current === X) === !!aIsX;
      var move;
      if (aToMove) {
        move = shapeAgentChoose(agentA, familyA, board, current, false, 0, depth);
      } else {
        move = shapeAgentChoose(agentB, familyB, board, current, false, 0, depth);
      }
      board = G.applyMove(board, current, move);
      if (G.findWinner(board) || G.isDraw(board)) {
        break;
      }
      current = G.other(current);
    }
    var markA = aIsX ? X : O;
    return G.resultForPlayer(G.findWinner(board), markA);
  }

  /**
   * Two deterministic 1-ply games, each agent starts once.
   * liveWins/savedWins count only wins (a 2-0 sweep decides keep-best).
   */
  function shapeHeadToHeadTwoGames(liveAgent, family, savedAgent, maxDepth) {
    var depth = typeof maxDepth === "number" && maxDepth > 0 ? maxDepth : 1;
    var asX = shapePlayAgents(liveAgent, family, savedAgent, family, true, depth);
    var asO = shapePlayAgents(liveAgent, family, savedAgent, family, false, depth);
    var liveWins = 0;
    var savedWins = 0;
    var draws = 0;
    if (asX === "win") {
      liveWins += 1;
    } else if (asX === "loss") {
      savedWins += 1;
    } else {
      draws += 1;
    }
    if (asO === "win") {
      liveWins += 1;
    } else if (asO === "loss") {
      savedWins += 1;
    } else {
      draws += 1;
    }
    return { liveWins: liveWins, savedWins: savedWins, draws: draws };
  }

  /** Eval vs uniform random, explore off, optional ply cap. No learning. */
  function shapeEvalVsRandom(agent, family, games, maxDepth, opts) {
    opts = opts || {};
    var wins = 0;
    var draws = 0;
    var losses = 0;
    var depth = typeof maxDepth === "number" && maxDepth > 0 ? maxDepth : 1;
    var extendTo = typeof opts.extendTo === "number" && opts.extendTo > games ?
      opts.extendTo : null;
    var extendIfWr = typeof opts.extendIfWr === "number" ? opts.extendIfWr : null;
    var target = Math.max(1, games | 0);
    var g = 0;
    while (g < target) {
      var learnerMark = Math.random() < 0.5 ? X : O;
      var board = G.emptyBoard();
      var current = X;
      while (true) {
        var moves = G.legalMoves(board);
        if (!moves.length) {
          break;
        }
        var move;
        if (current === learnerMark) {
          move = shapeAgentChoose(agent, family, board, current, false, 0, depth);
        } else {
          move = moves[Math.floor(Math.random() * moves.length)];
        }
        board = G.applyMove(board, current, move);
        if (G.findWinner(board) || G.isDraw(board)) {
          break;
        }
        current = G.other(current);
      }
      var result = G.resultForPlayer(G.findWinner(board), learnerMark);
      if (result === "win") {
        wins += 1;
      } else if (result === "draw") {
        draws += 1;
      } else {
        losses += 1;
      }
      g += 1;
      if (
        extendTo !== null &&
        extendIfWr !== null &&
        g === games &&
        (wins + 0.5 * draws) / g > extendIfWr &&
        target < extendTo
      ) {
        target = extendTo;
      }
    }
    return {
      wins: wins,
      draws: draws,
      losses: losses,
      games: g,
      winRate: (wins + 0.5 * draws) / Math.max(1, g),
      extended: g > games
    };
  }

  /**
   * Keep-best playoff: live vs saved. Explore off, 1-ply.
   * Returns "current" or "saved".
   */
  function shapeCompareKeepBest(liveAgent, family, savedAgent) {
    var api = global.C4_APP;
    var ply = 1;
    var n = 100;
    var h2h = api.shapeHeadToHeadTwoGames(liveAgent, family, savedAgent, ply);
    if (h2h.liveWins === 2) {
      return "current";
    }
    if (h2h.savedWins === 2) {
      return "saved";
    }
    var liveR = api.shapeEvalVsRandom(liveAgent, family, n, ply);
    var savedR = api.shapeEvalVsRandom(savedAgent, family, n, ply);
    var pLive = shapeEvalPoints(liveR);
    var pSaved = shapeEvalPoints(savedR);
    if (pLive > pSaved) {
      return "current";
    }
    if (pSaved > pLive) {
      return "saved";
    }
    var live1 = api.shapeEvalVsMinimaxTimed(liveAgent, family, 1, n, ply, null);
    var saved1 = api.shapeEvalVsMinimaxTimed(savedAgent, family, 1, n, ply, null);
    pLive = shapeEvalPoints(live1);
    pSaved = shapeEvalPoints(saved1);
    if (pLive > pSaved) {
      return "current";
    }
    if (pSaved > pLive) {
      return "saved";
    }
    if (pLive === 0 && pSaved === 0) {
      return "current";
    }
    var live2 = api.shapeEvalVsMinimaxTimed(liveAgent, family, 2, n, ply, null);
    var saved2 = api.shapeEvalVsMinimaxTimed(savedAgent, family, 2, n, ply, null);
    pLive = shapeEvalPoints(live2);
    pSaved = shapeEvalPoints(saved2);
    if (pSaved > pLive) {
      return "saved";
    }
    return "current";
  }

  function calibrateMinimaxMoveMs(level, samples) {
    var n = samples || 8;
    var opp = new M.FractionalMinimaxOpponent(M.normalizeLevel(level));
    var total = 0;
    var i;
    for (i = 0; i < n; i += 1) {
      var board = G.emptyBoard();
      /* Sprinkle a few random plies so positions vary. */
      var p = X;
      var r;
      for (r = 0; r < 4; r += 1) {
        var mv = G.legalMoves(board);
        if (!mv.length) {
          break;
        }
        board = G.applyMove(board, p, mv[Math.floor(Math.random() * mv.length)]);
        p = G.other(p);
      }
      var t0 = performance.now();
      opp.chooseMove(board, p);
      total += performance.now() - t0;
    }
    return Math.max(1, total / n);
  }

  /** Empty-board FractionalMinimax(level) think time, doubled for tournament parity. */
  function calibrateTournamentMoveMs(level) {
    if (!(typeof level === "number") || level <= 0) {
      return 2;
    }
    var opp = new M.FractionalMinimaxOpponent(M.normalizeLevel(level));
    var board = G.emptyBoard();
    var t0 = performance.now();
    opp.chooseMove(board, X);
    var one = Math.max(1, performance.now() - t0);
    return one * 2;
  }

  function shapeAgentToDict(agent, family) {
    if (agent && typeof agent.toDict === "function") {
      return { family: family, data: agent.toDict(), layerSizes: agent.net ? agent.net.layerSizes : (agent.layerSizes || null) };
    }
    return { family: family, data: null };
  }

  function shapeAgentFromDict(blob) {
    var family = blob.family;
    var agent = createShapeAgent(family, blob.layerSizes);
    if (blob.data && agent && typeof agent.constructor.fromDict === "function") {
      return agent.constructor.fromDict(blob.data);
    }
    if (blob.data && family.indexOf("tfjs") === 0) {
      return family === "tfjs_policy" ?
        TfjsPolicyAgent.fromDict(blob.data) :
        TfjsValueAgent.fromDict(blob.data);
    }
    return agent;
  }

  global.C4_APP = {
    init: init,
    displayName: displayName,
    playAlgoHasTimedSearch: playAlgoHasTimedSearch,
    PLAY_HUMAN: "human",
    Application: Application,
    formatErrorReport: formatErrorReport,
    nnExploreRate: nnExploreRate,
    sampleSoftmaxFromLogits: sampleSoftmaxFromLogits,
    pickPeerIndex: pickPeerIndex,
    gaFitnessGames: gaFitnessGames,
    getExecTrace: function () {
      return execTrace.slice();
    },
    SHAPE_FAMILIES: SHAPE_FAMILIES,
    createShapeAgent: createShapeAgent,
    defaultLayerSizes: defaultLayerSizes,
    defaultLearningRate: defaultLearningRate,
    applyShapeLearningRate: applyShapeLearningRate,
    shapeTrainVsRandom: shapeTrainVsRandom,
    shapeTrainVsSelf: shapeTrainVsSelf,
    shapeTrainVsMinimax: shapeTrainVsMinimax,
    shapeTrainVsPeer: shapeTrainVsPeer,
    shapeEvalPoints: shapeEvalPoints,
    shapePlayAgents: shapePlayAgents,
    shapeHeadToHeadTwoGames: shapeHeadToHeadTwoGames,
    shapeEvalVsRandom: shapeEvalVsRandom,
    shapeCompareKeepBest: shapeCompareKeepBest,
    shapeEvolveOne: shapeEvolveOne,
    shapeEvalVsMinimaxTimed: shapeEvalVsMinimaxTimed,
    shapeAgentChoose: shapeAgentChoose,
    botSearchDepthForLevel: botSearchDepthForLevel,
    calibrateMinimaxMoveMs: calibrateMinimaxMoveMs,
    calibrateTournamentMoveMs: calibrateTournamentMoveMs,
    shapeAgentToDict: shapeAgentToDict,
    shapeAgentFromDict: shapeAgentFromDict,
    NnAgent: NnAgent,
    NnValueAgent: NnValueAgent,
    ReinforceAgent: ReinforceAgent,
    ReinforceValueAgent: ReinforceValueAgent
  };
})(window);
