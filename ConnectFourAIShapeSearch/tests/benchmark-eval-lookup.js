/* Manual benchmark: current window eval vs (rw/yw) + POP4 + SCORE_BY_PO lookup.
   Fixed-depth alphabeta depths 3-10 from empty board. Not in default test suite.
   Run: node tests/benchmark-eval-lookup.js */
var path = require("path");

var root = path.join(__dirname, "..");
process.chdir(root);

global.window = global;
global.performance = { now: function () { return Date.now(); } };

require(path.join(root, "c4_js/constants.js"));
require(path.join(root, "c4_js/game.js"));
require(path.join(root, "c4_js/minimax.js"));

var G = global.C4_GAME;
var C = global.C4_CONSTANTS;
var M = global.C4_MINIMAX;

var WIN_SCORE = 1000000;
var CENTER_COL = 3;
var CENTER_BONUS = 6;
var CENTER_MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6];
var RUNS = 3;
var DEPTHS = [3, 4, 5, 6, 7, 8, 9, 10];

var POP4 = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

var SCORE_BY_PO = [
  [0, 0, -15, -120, 0],
  [1, 0, 0, 0, 0],
  [10, 0, 0, 0, 0],
  [100, 0, 0, 0, 0],
  [10000, 0, 0, 0, 0]
];

function buildLookupWindows() {
  var wins = [];
  var r;
  var c;
  function push4(b0, b1, b2, b3) {
    wins.push({
      W: b0 | b1 | b2 | b3,
      bits: [b0, b1, b2, b3]
    });
  }
  for (r = 0; r < C.ROWS; r += 1) {
    for (c = 0; c < C.COLS - 3; c += 1) {
      push4(
        G.cellBit(c, r),
        G.cellBit(c + 1, r),
        G.cellBit(c + 2, r),
        G.cellBit(c + 3, r)
      );
    }
  }
  for (c = 0; c < C.COLS; c += 1) {
    for (r = 0; r < C.ROWS - 3; r += 1) {
      push4(
        G.cellBit(c, r),
        G.cellBit(c, r + 1),
        G.cellBit(c, r + 2),
        G.cellBit(c, r + 3)
      );
    }
  }
  for (c = 0; c < C.COLS - 3; c += 1) {
    for (r = 0; r < C.ROWS - 3; r += 1) {
      push4(
        G.cellBit(c, r),
        G.cellBit(c + 1, r + 1),
        G.cellBit(c + 2, r + 2),
        G.cellBit(c + 3, r + 3)
      );
    }
  }
  for (c = 0; c < C.COLS - 3; c += 1) {
    for (r = 3; r < C.ROWS; r += 1) {
      push4(
        G.cellBit(c, r),
        G.cellBit(c + 1, r - 1),
        G.cellBit(c + 2, r - 2),
        G.cellBit(c + 3, r - 3)
      );
    }
  }
  return wins;
}

var LOOKUP_WINDOWS = buildLookupWindows();

var CENTER_BITS = [];
var cr;
for (cr = 0; cr < C.ROWS; cr += 1) {
  CENTER_BITS.push(G.cellBit(CENTER_COL, cr));
}

function encode4(rw, bits) {
  return (rw & bits[0] ? 1 : 0) |
    (rw & bits[1] ? 2 : 0) |
    (rw & bits[2] ? 4 : 0) |
    (rw & bits[3] ? 8 : 0);
}

function terminalScore(board, evalPlayer) {
  if (G.hasWin(board.red)) {
    return evalPlayer === C.X ? WIN_SCORE : -WIN_SCORE;
  }
  if (G.hasWin(board.yellow)) {
    return evalPlayer === C.O ? WIN_SCORE : -WIN_SCORE;
  }
  if (G.isBoardFull(board)) {
    return 0;
  }
  return null;
}

function evaluateLookup(board, player) {
  var terminal = terminalScore(board, player);
  if (terminal !== null) {
    return terminal;
  }
  var playerBits = player === C.X ? board.red : board.yellow;
  var oppBits = player === C.X ? board.yellow : board.red;
  var total = 0;
  var w;
  for (w = 0; w < LOOKUP_WINDOWS.length; w += 1) {
    var win = LOOKUP_WINDOWS[w];
    var rw = playerBits & win.W;
    var ow = oppBits & win.W;
    if (rw & ow) {
      continue;
    }
    var p = POP4[encode4(rw, win.bits)];
    var o = POP4[encode4(ow, win.bits)];
    if (p > 0 && o > 0) {
      continue;
    }
    total += SCORE_BY_PO[p][o];
  }
  for (w = 0; w < CENTER_BITS.length; w += 1) {
    if (playerBits & CENTER_BITS[w]) {
      total += CENTER_BONUS;
    }
  }
  return total;
}

function orderMovesCenterOut(moves) {
  var ordered = [];
  var i;
  var j;
  for (i = 0; i < CENTER_MOVE_ORDER.length; i += 1) {
    for (j = 0; j < moves.length; j += 1) {
      if (moves[j] === CENTER_MOVE_ORDER[i]) {
        ordered.push(moves[j]);
        break;
      }
    }
  }
  return ordered.length ? ordered : moves.slice();
}

function alphabetaWithEval(board, depth, alpha, beta, sideToMove, evalPlayer, evalFn) {
  var maximizing = sideToMove === evalPlayer;
  var terminal = terminalScore(board, evalPlayer);
  if (terminal !== null || depth === 0) {
    return { score: terminal !== null ? terminal : evalFn(board, evalPlayer), move: null };
  }
  var moves = orderMovesCenterOut(G.legalMoves(board));
  if (!moves.length) {
    return { score: evalFn(board, evalPlayer), move: null };
  }
  var bestMove = moves[0];
  if (maximizing) {
    var value = -Infinity;
    var i;
    var token;
    for (i = 0; i < moves.length; i += 1) {
      token = G.makeMove(board, sideToMove, moves[i]);
      if (!token) {
        continue;
      }
      var sc = alphabetaWithEval(
        board, depth - 1, alpha, beta, G.other(sideToMove), evalPlayer, evalFn
      ).score;
      G.undoMove(board, token);
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
    token = G.makeMove(board, sideToMove, moves[i]);
    if (!token) {
      continue;
    }
    sc = alphabetaWithEval(
      board, depth - 1, alpha, beta, G.other(sideToMove), evalPlayer, evalFn
    ).score;
    G.undoMove(board, token);
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

function runDepth(depth, evalFn) {
  var board = G.cloneBoard(G.emptyBoard());
  var t0 = performance.now();
  alphabetaWithEval(board, depth, -Infinity, Infinity, C.X, C.X, evalFn);
  return performance.now() - t0;
}

function median(nums) {
  var sorted = nums.slice().sort(function (a, b) { return a - b; });
  var mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

function benchLabel(evalFn, depth) {
  var times = [];
  var r;
  for (r = 0; r < RUNS; r += 1) {
    times.push(runDepth(depth, evalFn));
  }
  return median(times);
}

/* Sanity: scores must match on sample positions */
var samples = [
  G.emptyBoard(),
  G.applyMove(G.emptyBoard(), C.X, 3),
  G.applyMove(G.applyMove(G.emptyBoard(), C.X, 3), C.O, 2)
];
var s;
for (s = 0; s < samples.length; s += 1) {
  var a = M.evaluate(samples[s], C.X);
  var b = evaluateLookup(samples[s], C.X);
  if (a !== b) {
    console.log("FAIL: eval mismatch at sample " + s + ": current=" + a + " lookup=" + b);
    process.exit(1);
  }
}

console.log("Empty board, red to move, fixed-depth alphabeta, median of " + RUNS + " runs");
console.log("Sanity OK: lookup eval matches current eval on sample boards");
console.log("");
console.log("depth | current ms | lookup ms | speedup");
console.log("------+------------+-----------+--------");

var d;
for (d = 0; d < DEPTHS.length; d += 1) {
  var depth = DEPTHS[d];
  runDepth(depth, M.evaluate);
  runDepth(depth, evaluateLookup);
  var currentMs = benchLabel(M.evaluate, depth);
  var lookupMs = benchLabel(evaluateLookup, depth);
  var speedup = lookupMs > 0 ? (currentMs / lookupMs).toFixed(2) : "n/a";
  console.log(
    ("  " + depth).slice(-2) + "   | " +
    currentMs.toFixed(1).padStart(10) + " | " +
    lookupMs.toFixed(1).padStart(9) + " | " +
    speedup + "x"
  );
}
