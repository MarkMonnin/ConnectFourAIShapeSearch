/* Core game logic, minimax, and play API unit tests (Node). */
var fs = require("fs");
var vm = require("vm");
var path = require("path");

var root = path.join(__dirname, "..");
process.chdir(root);

global.window = global;
global.localStorage = {
  getItem: function () { return null; },
  setItem: function () {},
  removeItem: function () {}
};
global.performance = { now: function () { return Date.now(); } };

require(path.join(root, "c4_js/constants.js"));
require(path.join(root, "c4_js/game.js"));
require(path.join(root, "c4_js/minimax.js"));
vm.runInThisContext(fs.readFileSync(path.join(root, "c4-network.js"), "utf8"));
vm.runInThisContext(fs.readFileSync(path.join(root, "c4-opponents.js"), "utf8"));
require(path.join(root, "c4_js/tfjs-agent.js"));
require(path.join(root, "c4_js/engine.js"));
require(path.join(root, "c4_js/play.js"));

var G = global.C4_GAME;
var C = global.C4_CONSTANTS;
var M = global.C4_MINIMAX;
var APP = global.C4_APP;

var failures = [];

function fail(msg) {
  failures.push(msg);
}

function assert(cond, msg) {
  if (!cond) {
    fail(msg);
  }
}

function fillColumn(board, col, count, startPlayer) {
  var b = board;
  var p = startPlayer;
  var i;
  for (i = 0; i < count; i += 1) {
    b = G.applyMove(b, p, col);
    p = G.other(p);
  }
  return b;
}

function testMakeMoveUndoRoundtrip() {
  var b = G.cloneBoard(G.emptyBoard());
  var key0 = G.boardKey(b);
  var t1 = G.makeMove(b, C.X, 3);
  assert(t1 !== null, "makeMove returns token");
  assert(G.boardCellValue(b, 3, 0) === C.X, "makeMove sets bit");
  G.undoMove(b, t1);
  assert(G.boardKey(b) === key0, "undoMove restores board");
  var t2 = G.makeMove(b, C.O, 3);
  G.undoMove(b, t2);
  assert(G.boardKey(b) === key0, "undoMove restores after O");
}

function testIsBoardFull() {
  var b = G.emptyBoard();
  assert(!G.isBoardFull(b), "empty board not full");
  b = fillColumn(b, 0, C.ROWS, C.X);
  assert(!G.isBoardFull(b), "one column not full");
}

function testBigIntNoAliasing() {
  var b = G.applyMove(G.emptyBoard(), C.X, 3);
  assert(G.boardCellValue(b, 3, 0) === C.X, "bit at col3 row0");
  assert(!G.boardCellValue(b, 0, 5), "bit 3 must not alias col0 row5");
}

function buildVerticalWin(player, col) {
  var b = G.emptyBoard();
  var filler = 0;
  var i;
  for (i = 0; i < 4; i += 1) {
    while (filler === col) {
      filler = (filler + 1) % C.COLS;
    }
    b = G.applyMove(b, G.other(player), filler);
    b = G.applyMove(b, player, col);
    filler = (filler + 1) % C.COLS;
  }
  return b;
}

function testWinDetection() {
  var b = G.emptyBoard();
  var cols = [3, 3, 4, 4, 5, 5, 6];
  var i;
  for (i = 0; i < cols.length; i += 1) {
    b = G.applyMove(b, i % 2 === 0 ? C.X : C.O, cols[i]);
  }
  assert(G.findWinner(b) === C.X, "horizontal win for X");

  b = buildVerticalWin(C.O, 1);
  assert(G.findWinner(b) === C.O, "vertical win for O in col 1");
}

function testBoardKeyAndClone() {
  var a = G.applyMove(G.emptyBoard(), C.X, 2);
  var b = G.applyMove(a, C.O, 3);
  assert(G.boardKey(a) !== G.boardKey(b), "distinct positions have distinct keys");
  var c = G.cloneBoard(b);
  c = G.applyMove(c, C.X, 1);
  assert(G.boardKey(b) !== G.boardKey(c), "clone is independent");
}

function testBoardToInput() {
  var b = G.applyMove(G.emptyBoard(), C.X, 3);
  var inp = G.boardToInput(b, C.X);
  assert(inp.length === G.POLICY_INPUT_DIM, "policy input length cells+side+features");
  assert(inp[3] === 1, "own piece from X view at col3 row0");
  var red = G.boardToRedInput(b);
  assert(red.length === G.VALUE_INPUT_DIM, "value input length cells+features");
  assert(red[3] === 1, "red piece at col3 row0");
}

function testMirrorBoardAndTrajectory() {
  var b = G.applyMove(G.emptyBoard(), C.X, 0);
  b = G.applyMove(b, C.O, 1);
  assert(G.mirrorColumn(0) === 6, "mirror col 0 -> 6");
  assert(G.mirrorColumn(3) === 3, "center col mirrors to itself");
  var m = G.mirrorBoard(b);
  assert(G.boardCellValue(m, 6, 0) === C.X, "X at col0 mirrors to col6");
  assert(G.boardCellValue(m, 5, 0) === C.O, "O at col1 mirrors to col5");
  assert(G.boardKey(G.mirrorBoard(m)) === G.boardKey(b), "mirror twice is identity");
  var step = {
    board: G.cloneBoard(b),
    player: C.X,
    action: 0,
    input: G.boardToInput(b, C.X)
  };
  var expanded = G.expandTrajectoryWithMirrors([step]);
  assert(expanded.length === 2, "mirror expand doubles policy steps");
  assert(expanded[1].action === 6, "mirrored action");
  assert(expanded[1].input[6] === 1, "mirrored input has own piece at col6");
  var valueExpanded = G.expandTrajectoryWithMirrors([{ board: G.cloneBoard(b) }]);
  assert(valueExpanded.length === 2, "mirror expand doubles value steps");
  assert(G.boardKey(valueExpanded[1].board) === G.boardKey(m), "value mirror board");
}

function testNnMirrorAugmentsTrainSteps() {
  var app = new APP.Application();
  var agent = app.registry.nn2;
  var before = agent.trainSteps;
  var b = G.applyMove(G.emptyBoard(), C.X, 2);
  var step = agent.buildStep(b, C.X, 2);
  assert(!!step.board && step.player === C.X, "policy buildStep stores board+player");
  agent.learnFromTrajectory([step], C.X, C.X);
  assert(agent.trainSteps === before + 2, "nn learn uses original + mirror");
}

function testNnExploreSoftmaxAndDecay() {
  assert(Math.abs(APP.nnExploreRate(0) - 0.25) < 1e-9, "explore start 0.25");
  assert(Math.abs(APP.nnExploreRate(3000) - 0.05) < 1e-9, "explore end 0.05 at 3000");
  assert(APP.nnExploreRate(1500) > 0.14 && APP.nnExploreRate(1500) < 0.16, "explore mid ~0.15");
  assert(APP.nnExploreRate(10000) === APP.nnExploreRate(3000), "explore clamped at min");
  var outputs = [0, 0, 0, 10, 0, 0, 0];
  var picks = {};
  var i;
  for (i = 0; i < 80; i += 1) {
    var m = APP.sampleSoftmaxFromLogits(outputs, [0, 1, 2, 3, 4, 5, 6], 1);
    picks[m] = (picks[m] || 0) + 1;
  }
  assert((picks[3] || 0) >= 50, "softmax prefers high logit col 3");
  assert(APP.sampleSoftmaxFromLogits(outputs, [3], 1) === 3, "single legal move");
}

function testNoOneLayerBots() {
  var ones = ["nn1", "nn_value1", "reinforce1", "reinforce_value1"];
  var i;
  for (i = 0; i < ones.length; i += 1) {
    assert(C.TRAINING_ALGO_IDS.indexOf(ones[i]) < 0, ones[i] + " removed from training");
    assert(!C.ALGO_NAMES[ones[i]], ones[i] + " removed from names");
  }
  var app = new APP.Application();
  for (i = 0; i < ones.length; i += 1) {
    assert(!app.registry[ones[i]], ones[i] + " not in registry");
  }
  assert(!!app.registry.nn2 && !!app.registry.nn3, "2L/3L backprop remain");
}

function testEvaluateTerminal() {
  var b = G.emptyBoard();
  var cols = [0, 1, 2, 3];
  var i;
  for (i = 0; i < 4; i += 1) {
    b = G.applyMove(b, C.X, cols[i]);
    if (i < 3) {
      b = G.applyMove(b, C.O, 6);
    }
  }
  assert(G.findWinner(b) === C.X, "setup win");
  assert(M.evaluate(b, C.X) > 900000, "eval win score for X");
  assert(M.evaluate(b, C.O) < -900000, "eval loss score for O");
}

function testIsDraw() {
  assert(!G.isDraw(G.emptyBoard()), "empty board not draw");
  var b = G.applyMove(G.emptyBoard(), C.X, 3);
  assert(!G.isDraw(b), "partial board not draw");
  assert(G.isDraw(b) === (G.legalMoves(b).length === 0 && !G.findWinner(b)), "isDraw matches moves+winner");
}

function testAlphabetaLegalMove() {
  var b = G.emptyBoard();
  var res = M.alphabeta(b, 2, -Infinity, Infinity, C.X, C.X, Math.random);
  var moves = G.legalMoves(b);
  assert(moves.indexOf(res.move) >= 0, "alphabeta returns legal move");
}

function testChooseMoveUntilDeadlineDeepens() {
  var start = performance.now();
  var deadline = start + 3000;
  var timed = M.chooseMoveUntilDeadline(G.emptyBoard(), C.X, deadline, Math.random);
  assert(typeof timed.move === "number", "timed minimax returns move");
  assert(timed.depth > 7, "3s budget on empty board should exceed depth 7, got " + timed.depth);
  assert(G.legalMoves(G.emptyBoard()).indexOf(timed.move) >= 0, "timed move legal");
}

function testChooseMoveUntilDeadlineScalesWithBudget() {
  var shallow = M.chooseMoveUntilDeadline(
    G.emptyBoard(), C.X, performance.now() + 100, Math.random);
  var deep = M.chooseMoveUntilDeadline(
    G.emptyBoard(), C.X, performance.now() + 3000, Math.random);
  assert(shallow.depth >= 1, "short budget completes at least depth 1");
  assert(deep.depth > shallow.depth,
    "longer budget should search deeper (" + deep.depth + " vs " + shallow.depth + ")");
  assert(deep.depth > 7, "3s budget should pass old depth-7 cap, got " + deep.depth);
}

function testChooseMoveUntilDeadlineRespectsBudget() {
  var budgetMs = 1000;
  var t0 = performance.now();
  M.chooseMoveUntilDeadline(G.emptyBoard(), C.X, t0 + budgetMs, Math.random);
  var elapsed = performance.now() - t0;
  assert(elapsed < budgetMs + 250,
    "1s budget should stop near 1s, took " + elapsed.toFixed(0) + "ms");
}

function testValueMinimaxTimedRespectsBudget() {
  var app = new C4_APP.Application(true);
  var budgetMs = 1000;
  var t0 = performance.now();
  app.registry.reinforce_value3.chooseMoveTimed(G.emptyBoard(), C.X, budgetMs);
  var elapsed = performance.now() - t0;
  assert(elapsed < budgetMs + 250,
    "REINFORCE value timed search should stop near 1s, took " + elapsed.toFixed(0) + "ms");
}

function testDropRowAndLegalMoves() {
  var b = G.emptyBoard();
  assert(G.dropRow(b, 3) === 0, "empty col drops to row 0");
  assert(G.legalMoves(b).length === C.COLS, "all columns legal on empty board");
  b = fillColumn(b, 2, C.ROWS, C.X);
  assert(G.legalMoves(b).indexOf(2) < 0, "full column not legal");
  assert(G.dropRow(b, 2) < 0, "full column has no drop row");
}

function testApplyMoveFullColumnNoOp() {
  var b = fillColumn(G.emptyBoard(), 5, C.ROWS, C.X);
  var key = G.boardKey(b);
  b = G.applyMove(b, C.O, 5);
  assert(G.boardKey(b) === key, "move into full column leaves board unchanged");
}

function testOtherPlayer() {
  assert(G.other(C.X) === C.O, "other of X is O");
  assert(G.other(C.O) === C.X, "other of O is X");
}

function testPickPlayMoveLegal() {
  var app = new C4_APP.Application(true);
  var b = G.emptyBoard();
  var ids = ["qtable", "minimax", "mcts", "nn2"];
  var i;
  for (i = 0; i < ids.length; i += 1) {
    var col = app.pickPlayMove(ids[i], b, C.X, ids[i] === "minimax" ? 0.1 : 0);
    assert(G.legalMoves(b).indexOf(col) >= 0, ids[i] + " pickPlayMove legal");
  }
}

function testPlayAlgoTimedFlags() {
  assert(APP.playAlgoHasTimedSearch("minimax"), "minimax timed");
  assert(APP.playAlgoHasTimedSearch("mcts"), "mcts timed");
  assert(APP.playAlgoHasTimedSearch("nn2"), "nn timed");
  assert(!APP.playAlgoHasTimedSearch("qtable"), "qtable instant");
  assert(!APP.playAlgoHasTimedSearch("menace"), "menace instant");
}

function testHasWinBitboard() {
  var bits = G.cellBit(4, 0) | G.cellBit(4, 1) | G.cellBit(4, 2) | G.cellBit(4, 3);
  assert(G.hasWin(bits), "four stacked is vertical win in col");
  assert(!G.hasWin(G.cellBit(4, 0) | G.cellBit(4, 1) | G.cellBit(4, 2)), "three stacked is not a win");
  assert(G.hasWin(G.cellBit(0, 0) | G.cellBit(1, 0) | G.cellBit(2, 0) | G.cellBit(3, 0)), "horizontal four in row 0");
  assert(G.hasWin(G.cellBit(0, 0) | G.cellBit(1, 1) | G.cellBit(2, 2) | G.cellBit(3, 3)), "main diagonal");
  assert(G.hasWin(G.cellBit(3, 0) | G.cellBit(2, 1) | G.cellBit(1, 2) | G.cellBit(0, 3)), "anti diagonal");
  /* Wrap false positives from Observe games (bits aligned across row edges). */
  var wrapDiag6 = G.cellBit(2, 0) | G.cellBit(1, 1) | G.cellBit(0, 2) | G.cellBit(6, 2);
  assert(!G.hasWin(wrapDiag6), "diag6 must not wrap col0 into col6");
  var wrapHoriz = G.cellBit(5, 0) | G.cellBit(6, 0) | G.cellBit(0, 1) | G.cellBit(1, 1);
  assert(!G.hasWin(wrapHoriz), "horizontal must not wrap across rows");
}

function testReloadResetsSliceScale() {
  var app = new APP.Application();
  app.training.sliceScale = 5;
  app.training.beginTrainingRound();
  var n = app.training.trainingEligibleCount();
  app.applyLoaded({
    history: [],
    rotation: 0,
    tournamentNumber: 0,
    dropsAvailable: 0,
    latestResult: null,
    tournamentMoveBudgetMs: null,
    trainingSliceScale: 5,
    loadError: null
  });
  assert(app.training.sliceScale === 1, "reload resets slice scale");
  assert(n > 0, "trainees exist");
  assert(Math.abs(app.training.roundSliceLimit - C.TOURNAMENT_INTERVAL / n) < 0.02,
    "reload recomputes slice from trainee count");
}

function testAdjustTrainingSliceScale() {
  var app = new APP.Application();
  var te = app.training;
  te.sliceScale = 1;
  te.adjustSliceScaleFromRound(40);
  assert(Math.abs(te.sliceScale - 0.875) < 0.001,
    "halfway toward shorter scale when round too long");
  te.sliceScale = 1;
  te.adjustSliceScaleFromRound(20);
  assert(Math.abs(te.sliceScale - 1.25) < 0.001,
    "halfway toward longer scale when round too short");
}

function testTrainingSessionSliceBudget() {
  var app = new APP.Application();
  var te = app.training;
  te.sliceScale = 1;
  te.beginTrainingRound();
  var n = te.roundActiveIds.length;
  assert(n > 0, "training round has bots");
  var fixed = te.roundSliceLimit;
  assert(Math.abs(fixed * n - C.TOURNAMENT_INTERVAL) < 0.02,
    "fixed slice times n bots targets 30s");
  var i;
  for (i = 0; i < n; i += 1) {
    assert(Math.abs(te.roundSliceLimit - fixed) < 0.001, "slice limit unchanged during session");
    te.sliceTrainingElapsed = te.roundSliceLimit;
    te.finishCurrentSlice();
  }
  assert(te.trainingRoundComplete, "one slice per bot completes round");
}

function testTrainingRoundDefersTournament() {
  var app = new APP.Application();
  var te = app.training;
  te.trainingRoundComplete = false;
  assert(!te.shouldStartTournament(), "mid-round does not start tournament");
  te.trainingRoundComplete = true;
  assert(te.shouldStartTournament(), "tournament after one full training rotation");
}

function testOneTrainingRotationEndsAtLastBot() {
  var app = new APP.Application();
  var te = app.training;
  te.beginTrainingRound();
  assert(te.roundActiveIds.length > 0, "training bots exist");
  te.roundBotIndex = te.roundActiveIds.length - 1;
  te.currentAlgoIndex = C.TRAINING_ALGO_IDS.indexOf(te.roundActiveIds[te.roundBotIndex]);
  te.sliceTrainingElapsed = te.roundSliceLimit + 0.1;
  var last = te.roundCurrentAlgoId();
  te.runOneStep();
  assert(te.trainingRoundComplete, "training round completes at last bot");
  assert(te.roundBotIndex >= te.roundActiveIds.length, "does not start second rotation");
  assert(te.shouldStartTournament(), "tournament follows one rotation");
  assert(te.roundCurrentAlgoId() === last, "stays on last bot after round ends");
}

function testTfjsAgentRegistered() {
  var app = new APP.Application();
  assert(!!app.registry.tfjs, "tfjs agent registered");
  assert(!!app.registry.tfjs_value, "tfjs_value agent registered");
  assert(!!app.registry.tfjs_value5, "tfjs_value5 agent registered");
  assert(C.TRAINING_ALGO_IDS.indexOf("tfjs") >= 0, "tfjs in TRAINING_ALGO_IDS");
  assert(C.TRAINING_ALGO_IDS.indexOf("tfjs_value") >= 0, "tfjs_value in TRAINING_ALGO_IDS");
  assert(C.TRAINING_ALGO_IDS.indexOf("tfjs_value5") >= 0, "tfjs_value5 in TRAINING_ALGO_IDS");
  assert(C.VALUE_AGENT_IDS.indexOf("tfjs_value") >= 0, "tfjs_value in VALUE_AGENT_IDS");
  assert(C.VALUE_AGENT_IDS.indexOf("tfjs_value5") >= 0, "tfjs_value5 in VALUE_AGENT_IDS");
  assert(C.ALGO_NAMES.tfjs === "TF.js hybrid", "tfjs hybrid display name");
  assert(C.ALGO_NAMES.tfjs_value === "TF.js value 2L", "tfjs_value display name");
  assert(C.ALGO_NAMES.tfjs_value5 === "TF.js value 5L", "tfjs_value5 display name");
  assert(global.C4_TFJS.LEARNING_RATE === 0.001, "tfjs Adam default LR");
  assert(global.C4_TFJS.resolveAdamLearningRate(0.03) === 0.001, "migrate SGD 0.03 -> Adam 0.001");
  assert(global.C4_TFJS.resolveAdamLearningRate(0.0005) === 0.0005, "keep low Adam LR");
  assert(
    app.registry.tfjs_value5.layerSizes.join(",") === C.TFJS_VALUE_5L_SIZES.join(","),
    "tfjs_value5 uses wide 5-hidden sizes"
  );
  assert(app.registry.tfjs_value5.layerSizes.length === 7, "5 hidden => 7 size entries");
  var agent = app.registry.tfjs;
  if (agent.net.backend === "tfjs") {
    assert(agent.net.learningRate === 0.001, "tfjs hybrid uses Adam LR");
  }
  assert(typeof agent.backend === "function", "tfjs backend()");
  var b = G.emptyBoard();
  var col = agent.chooseMove(b, C.X, false);
  assert(G.legalMoves(b).indexOf(col) >= 0, "tfjs chooseMove legal");
  assert(typeof agent.net.getOutputsBatch === "function", "tfjs getOutputsBatch");
  assert(typeof agent.net.getValueBatch === "function", "tfjs getValueBatch");
  assert(typeof agent.net.probRedFromBoard === "function", "tfjs probRedFromBoard");
  var outs = agent.net.getOutputsBatch([
    G.boardToInput(b, C.X),
    G.boardToInput(b, C.X)
  ]);
  assert(outs.length === 2 && outs[0].length === 7, "batch forward shape");
  var vals = agent.net.getValueBatch([
    G.boardToInput(b, C.X),
    G.boardToInput(b, C.X)
  ]);
  assert(vals.length === 2, "value batch length");
  assert(vals[0] >= 0 && vals[0] <= 1, "value in [0,1]");
  var pRed = agent.net.probRedFromBoard(b);
  assert(pRed >= 0 && pRed <= 1, "probRedFromBoard in [0,1]");
  assert(!!agent.toDict().net.hasValueHead, "checkpoint marks value head");
  agent.learnFromTrajectory(
    [{
      input: G.boardToInput(b, C.X),
      action: col,
      board: G.cloneBoard(b),
      player: C.X
    }],
    C.X,
    C.X
  );
  assert(agent.trainSteps >= 2, "tfjs learns from trajectory + mirror");
  var dict = agent.toDict();
  var restored = global.C4_TFJS.TfjsPolicyAgent.fromDict(dict);
  assert(restored.trainSteps === agent.trainSteps, "tfjs save/load trainSteps");
  assert(typeof restored.net.probRedFromBoard === "function", "restored has value head");

  var vAgent = app.registry.tfjs_value;
  var vCol = vAgent.chooseMove(b, C.X, false);
  assert(G.legalMoves(b).indexOf(vCol) >= 0, "tfjs_value chooseMove legal");
  assert(typeof vAgent.net.getValueBatch === "function", "tfjs_value getValueBatch");
  var vBefore = vAgent.trainSteps;
  vAgent.learnFromTrajectory([{ board: G.cloneBoard(b) }], C.X, C.X);
  assert(vAgent.trainSteps === vBefore + 2, "tfjs_value learns board + mirror");
  var vDict = vAgent.toDict();
  var vRestored = global.C4_TFJS.TfjsValueAgent.fromDict(vDict);
  assert(vRestored.trainSteps === vAgent.trainSteps, "tfjs_value save/load trainSteps");
  assert(vRestored.probRedWins(b) >= 0 && vRestored.probRedWins(b) <= 1, "tfjs_value probRedWins");

  var v5 = app.registry.tfjs_value5;
  var v5Col = v5.chooseMove(b, C.X, false);
  assert(G.legalMoves(b).indexOf(v5Col) >= 0, "tfjs_value5 chooseMove legal");
  var v5Before = v5.trainSteps;
  v5.learnFromTrajectory([{ board: G.cloneBoard(b) }], C.X, C.X);
  assert(v5.trainSteps === v5Before + 2, "tfjs_value5 learns board + mirror");

  var tdNet = {
    probRedFromBoard: function () { return 0.25; }
  };
  var tdTraj = [
    { board: G.cloneBoard(b) },
    { board: G.applyMove(G.cloneBoard(b), C.X, 3) }
  ];
  var tdPairs = global.C4_TFJS.buildTdValuePairs(tdNet, tdTraj, C.X);
  assert(tdPairs.length === 4, "TD pairs = steps * 2 mirrors");
  assert(tdPairs[0].target !== 1, "first step mixes bootstrap not pure win");
  assert(tdPairs[2].target === 1, "last step uses outcome only");
  assert(tdPairs[0].target === tdPairs[1].target, "mirror keeps same TD target");

  var timedCol = vAgent.chooseMoveTimed(b, C.X, 25);
  assert(G.legalMoves(b).indexOf(timedCol) >= 0, "tfjs_value timed batched legal");
  assert(vAgent.lastDepth > 0, "tfjs_value timed sets lastDepth");
  timedCol = v5.chooseMoveTimed(b, C.X, 25);
  assert(G.legalMoves(b).indexOf(timedCol) >= 0, "tfjs_value5 timed batched legal");
  assert(v5.lastDepth > 0, "tfjs_value5 timed sets lastDepth");
  timedCol = agent.chooseMoveTimed(b, C.X, 25);
  assert(G.legalMoves(b).indexOf(timedCol) >= 0, "tfjs hybrid timed batched legal");
  assert(agent.lastDepth > 0, "tfjs hybrid timed sets lastDepth");
}

function testTabularLruCapReached() {
  var OPP = global.C4_OPPONENTS;
  var q = new OPP.QBrain();
  assert(!q.storageCapReached(), "fresh Q brain not cap reached");
  var i;
  for (i = 0; i < OPP.Q_MAX_STATES; i += 1) {
    q.ensureQ("k" + i + "|X");
  }
  assert(q.stateCount() === OPP.Q_MAX_STATES, "Q brain fills to hard cap");
  assert(!q.storageCapReached(), "cap flag not set until LRU evicts");
  q.ensureQ("kTrigger|X");
  assert(q.storageCapReached(), "Q brain sets cap flag when LRU evicts");
  assert(q.stateCount() < OPP.Q_MAX_STATES, "LRU keeps Q count below hard cap");
}

function testTabularTrainingDoneLegacyInfer() {
  var app = new APP.Application();
  app.registry.states.qtable.gamesPlayed = C.TABULAR_MAX_TRAINING_GAMES;
  app.registry.states.sarsa.gamesPlayed = C.TABULAR_MAX_TRAINING_GAMES - 1;
  var low = tabularLruLowWatermark(C.Q_MAX_STATES);
  var i;
  for (i = 0; i < low; i += 1) {
    app.registry.qtable.ensureQ("legacy" + i + "|X");
  }
  assert(app.training.orderedActiveIds().indexOf("qtable") < 0,
    "qtable with 10k games and post-LRU state count stops training");
  assert(app.training.orderedActiveIds().indexOf("sarsa") >= 0,
    "sarsa still trains below games cap");
}

function tabularLruLowWatermark(cap) {
  var OPP = global.C4_OPPONENTS;
  var batch = Math.max(1, Math.floor(cap * (OPP.EVICT_FRACTION || 0.1)));
  return cap - batch;
}

function testMaskAndCellBitUnique() {
  var a = G.cellBit(0, 5);
  var b = G.cellBit(3, 0);
  assert(a !== b, "cell bits distinct");
  assert((a & b) === 0n, "cell bits non-overlapping");
}

testMakeMoveUndoRoundtrip();
testIsBoardFull();
testBigIntNoAliasing();
testWinDetection();
testBoardKeyAndClone();
testBoardToInput();
testMirrorBoardAndTrajectory();
testNnMirrorAugmentsTrainSteps();
testNnExploreSoftmaxAndDecay();
testNoOneLayerBots();
testEvaluateTerminal();
testIsDraw();
testAlphabetaLegalMove();
testChooseMoveUntilDeadlineDeepens();
testChooseMoveUntilDeadlineScalesWithBudget();
testChooseMoveUntilDeadlineRespectsBudget();
testValueMinimaxTimedRespectsBudget();
testDropRowAndLegalMoves();
testApplyMoveFullColumnNoOp();
testOtherPlayer();
testPickPlayMoveLegal();
testPlayAlgoTimedFlags();
testHasWinBitboard();
function testGaPeerFitness() {
  assert(C.NEAT_FITNESS_GAMES === 10, "GA fitness uses 10 games");
  assert(APP.gaFitnessGames() === 10, "gaFitnessGames helper");
  var seen = {};
  var i;
  for (i = 0; i < 40; i += 1) {
    var p = APP.pickPeerIndex(5, 2);
    assert(p !== 2, "peer index avoids self");
    assert(p >= 0 && p < 5, "peer index in range");
    seen[p] = true;
  }
  assert(Object.keys(seen).length >= 2, "peer picks vary");
  var app = new APP.Application();
  assert(app.registry.neat.extraStats().indexOf("best=") >= 0, "neat shows best=");
  assert(app.registry.genetic_menace.extraStats().indexOf("best=") >= 0, "genetic shows best=");
}

testReloadResetsSliceScale();
testAdjustTrainingSliceScale();
testTrainingSessionSliceBudget();
testTrainingRoundDefersTournament();
testOneTrainingRotationEndsAtLastBot();
testTfjsAgentRegistered();
testGaPeerFitness();
testTabularLruCapReached();
testTabularTrainingDoneLegacyInfer();
testMaskAndCellBitUnique();

if (failures.length) {
  console.log("FAIL (" + failures.length + "):");
  var f;
  for (f = 0; f < failures.length; f += 1) {
    console.log("  " + failures[f]);
  }
  process.exit(1);
}

console.log("OK: game core + minimax timed deepening tests");
