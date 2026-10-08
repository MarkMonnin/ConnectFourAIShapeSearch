/* Manual perf benchmark only (NOT in safe-code.ps1 / default test suite).
   Depth-10 empty-board: make/undo vs applyMove. Results documented in c4_js/game.js.
   Run: node --expose-gc tests/benchmark-minimax-make-undo.js */
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

var G = global.C4_GAME;
var C = global.C4_CONSTANTS;
var M = global.C4_MINIMAX;

var DEPTH = 10;
var RUNS = 7;

function alphabetaAllocate(board, depth, alpha, beta, sideToMove, evalPlayer) {
  var maximizing = sideToMove === evalPlayer;
  var score = M.evaluate(board, evalPlayer);
  var terminal = G.hasWin(board.red) || G.hasWin(board.yellow) || G.isBoardFull(board);
  if (terminal || depth === 0) {
    return { score: score, move: null };
  }

  var moves = G.legalMoves(board);
  if (!moves.length) {
    return { score: score, move: null };
  }

  var bestMove = moves[0];
  if (maximizing) {
    var value = -Infinity;
    var i;
    for (i = 0; i < moves.length; i += 1) {
      var child = G.applyMove(board, sideToMove, moves[i]);
      var sc = alphabetaAllocate(child, depth - 1, alpha, beta, G.other(sideToMove), evalPlayer).score;
      if (sc > value) {
        value = sc;
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
    child = G.applyMove(board, sideToMove, moves[i]);
    sc = alphabetaAllocate(child, depth - 1, alpha, beta, G.other(sideToMove), evalPlayer).score;
    if (sc < value) {
      value = sc;
      bestMove = moves[i];
    }
    beta = Math.min(beta, value);
    if (beta <= alpha) {
      break;
    }
  }
  return { score: value, move: bestMove };
}

function runMakeUndo() {
  var board = G.cloneBoard(G.emptyBoard());
  return M.alphabeta(board, DEPTH, -Infinity, Infinity, C.X, C.X, Math.random);
}

function runAllocate() {
  return alphabetaAllocate(G.emptyBoard(), DEPTH, -Infinity, Infinity, C.X, C.X);
}

function median(nums) {
  var sorted = nums.slice().sort(function (a, b) { return a - b; });
  var mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

function bench(label, fn) {
  var times = [];
  var i;
  var last = null;
  for (i = 0; i < RUNS; i += 1) {
    var t0 = performance.now();
    last = fn();
    times.push(performance.now() - t0);
  }
  var sum = 0;
  for (i = 0; i < times.length; i += 1) {
    sum += times[i];
  }
  console.log(label + ":");
  console.log("  move=" + last.move + " score=" + last.score);
  console.log("  ms per run: " + times.map(function (t) { return t.toFixed(1); }).join(", "));
  console.log("  median=" + median(times).toFixed(1) + "ms mean=" + (sum / times.length).toFixed(1) + "ms");
  return median(times);
}

var sanityA = runMakeUndo();
var sanityB = runAllocate();
if (sanityA.move !== sanityB.move || sanityA.score !== sanityB.score) {
  console.log("FAIL: make/undo and allocate disagree");
  console.log("  make/undo: move=" + sanityA.move + " score=" + sanityA.score);
  console.log("  allocate:  move=" + sanityB.move + " score=" + sanityB.score);
  process.exit(1);
}

console.log("Depth " + DEPTH + " empty board, player X, " + RUNS + " runs each");
console.log("Sanity OK: both pick column " + (sanityA.move + 1) + " score " + sanityA.score);
console.log("");

if (global.gc) {
  global.gc();
}

var makeUndoMs = bench("make/undo (in-place)", runMakeUndo);

if (global.gc) {
  global.gc();
}

var allocateMs = bench("allocate-new (applyMove)", runAllocate);

var ratio = allocateMs / makeUndoMs;
console.log("");
console.log("allocate / make-undo median ratio: " + ratio.toFixed(2) + "x");
if (makeUndoMs < allocateMs) {
  console.log("make/undo is faster by " + ((1 - makeUndoMs / allocateMs) * 100).toFixed(0) + "% (median)");
} else {
  console.log("allocate-new was faster by " + ((1 - allocateMs / makeUndoMs) * 100).toFixed(0) + "% (median)");
}
