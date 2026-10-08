/* Minimax must block an immediate win threat for either side (regression). */
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

var G = global.C4_GAME;
var C = global.C4_CONSTANTS;
var M = global.C4_MINIMAX;

var failures = [];

function fail(msg) {
  failures.push(msg);
}

function assert(cond, msg) {
  if (!cond) {
    fail(msg);
  }
}

function markLabel(mark) {
  return mark === C.X ? "red" : "yellow";
}

/** User rows are top-to-bottom; game row 0 is bottom. */
function boardFromAscii(rowsTopToBottom) {
  var b = G.emptyBoard();
  var visualRow;
  var col;
  var ch;
  var mark;
  var cells = [];
  for (visualRow = 0; visualRow < rowsTopToBottom.length; visualRow += 1) {
    var line = rowsTopToBottom[visualRow];
    for (col = 0; col < C.COLS; col += 1) {
      ch = line.charAt(col);
      if (ch === "R") {
        mark = C.X;
      } else if (ch === "Y") {
        mark = C.O;
      } else {
        continue;
      }
      cells.push({ col: col, row: C.ROWS - 1 - visualRow, mark: mark });
    }
  }
  cells.sort(function (a, b) {
    if (a.col !== b.col) {
      return a.col - b.col;
    }
    return a.row - b.row;
  });
  var i;
  for (i = 0; i < cells.length; i += 1) {
    b = G.applyMove(b, cells[i].mark, cells[i].col);
  }
  return b;
}

function flipAsciiRows(rowsTopToBottom) {
  var out = [];
  var r;
  var line;
  var c;
  var ch;
  for (r = 0; r < rowsTopToBottom.length; r += 1) {
    line = rowsTopToBottom[r];
    var flipped = "";
    for (c = 0; c < line.length; c += 1) {
      ch = line.charAt(c);
      if (ch === "R") {
        flipped += "Y";
      } else if (ch === "Y") {
        flipped += "R";
      } else {
        flipped += ch;
      }
    }
    out.push(flipped);
  }
  return out;
}

function winningColumns(board, attacker) {
  var cols = [];
  var i;
  var moves = G.legalMoves(board);
  for (i = 0; i < moves.length; i += 1) {
    var c = moves[i];
    var next = G.applyMove(board, attacker, c);
    if (G.findWinner(next) === attacker) {
      cols.push(c);
    }
  }
  return cols;
}

function blockingColumns(board, defender) {
  var attacker = G.other(defender);
  var cols = [];
  var i;
  var moves = G.legalMoves(board);
  for (i = 0; i < moves.length; i += 1) {
    var c = moves[i];
    var afterDef = G.applyMove(board, defender, c);
    if (G.findWinner(afterDef) === defender) {
      cols.push(c);
      continue;
    }
    var stillThreat = false;
    var m;
    var oppMoves = G.legalMoves(afterDef);
    for (m = 0; m < oppMoves.length; m += 1) {
      var afterAtk = G.applyMove(afterDef, attacker, oppMoves[m]);
      if (G.findWinner(afterAtk) === attacker) {
        stillThreat = true;
        break;
      }
    }
    if (!stillThreat) {
      cols.push(c);
    }
  }
  return cols;
}

function runBlockScenario(rows, label, defender) {
  var attacker = G.other(defender);
  var board = boardFromAscii(rows);
  var winCols = winningColumns(board, attacker);
  var blocks = blockingColumns(board, defender);

  console.log("--- " + label + " ---");
  console.log(markLabel(attacker) + " immediate win columns (1-index):",
    winCols.map(function (c) { return c + 1; }).join(", ") || "(none)");
  console.log(markLabel(defender) + " blocking columns (1-index):",
    blocks.map(function (c) { return c + 1; }).join(", ") || "(none)");

  assert(winCols.length > 0, label + ": should have " + markLabel(attacker) + " winning threat");
  assert(blocks.length > 0, label + ": should have " + markLabel(defender) + " blocking move");

  var res1 = M.alphabeta(board, 1, -Infinity, Infinity, defender, defender, Math.random);
  assert(blocks.indexOf(res1.move) >= 0,
    label + " depth 1 should block, chose col " + (res1.move + 1));

  var timed = M.chooseMoveUntilDeadline(board, defender, performance.now() + 500, Math.random);
  assert(blocks.indexOf(timed.move) >= 0,
    label + " timed search should block, chose col " + (timed.move + 1));

  var app = new C4_APP.Application(true);
  var playCol = app.pickPlayMove("minimax", board, defender, 1);
  assert(blocks.indexOf(playCol) >= 0,
    label + " pickPlayMove should block, chose col " + (playCol + 1));
}

var YELLOW_DEFENDS_ROWS = [
  "bbbbbbb",
  "bYRYbbb",
  "YRRYYbb",
  "RYYRRYb",
  "YRRYYRb",
  "RRYRRRY"
];

runBlockScenario(YELLOW_DEFENDS_ROWS, "yellow blocks red win in column 4", C.O);

var afterRedWins = boardFromAscii([
  "bbbRbbb",
  "bYRYYbb",
  "YRRYYbb",
  "RYYRRYb",
  "YRRYYRb",
  "RRYRRRY"
]);
assert(G.findWinner(afterRedWins) === C.X, "after red takes col 4, red should have won");

var RED_DEFENDS_ROWS = flipAsciiRows(YELLOW_DEFENDS_ROWS);
runBlockScenario(RED_DEFENDS_ROWS, "red blocks yellow win in column 4 (colors flipped)", C.X);

var afterYellowWins = boardFromAscii(flipAsciiRows([
  "bbbRbbb",
  "bYRYYbb",
  "YRRYYbb",
  "RYYRRYb",
  "YRRYYRb",
  "RRYRRRY"
]));
assert(G.findWinner(afterYellowWins) === C.O, "after yellow takes col 4, yellow should have won");

function testTrainingMinimaxOpponent(defender, rows, label) {
  var board = boardFromAscii(rows);
  var blocks = blockingColumns(board, defender);
  var rng = M.makeSeededRng(42);
  var depth1 = new M.MinimaxOpponent(1, rng);
  var move = depth1.chooseMove(board, defender);
  assert(blocks.indexOf(move) >= 0,
    label + " MinimaxOpponent depth 1 blocks, chose col " + (move + 1));

  var frac = new M.FractionalMinimaxOpponent(1, rng);
  move = frac.chooseMove(board, defender);
  assert(blocks.indexOf(move) >= 0,
    label + " FractionalMinimaxOpponent level 1 blocks, chose col " + (move + 1));

  var fracDeep = new M.FractionalMinimaxOpponent(2.7, rng);
  move = fracDeep.chooseMove(board, defender);
  assert(blocks.indexOf(move) >= 0,
    label + " FractionalMinimaxOpponent level 2.7 blocks, chose col " + (move + 1));
}

console.log("--- training minimax opponents ---");
testTrainingMinimaxOpponent(C.O, YELLOW_DEFENDS_ROWS, "yellow");
testTrainingMinimaxOpponent(C.X, RED_DEFENDS_ROWS, "red");

console.log("--- fractional depth cap + threat ordering ---");
assert(M.MAX_FRACTIONAL_DEPTH === 20, "MAX_FRACTIONAL_DEPTH is 20");
var depthRng = M.makeSeededRng(1);
assert(M.chooseDepthForMove(20, depthRng) <= 20, "level 20 capped at 20");
assert(M.chooseDepthForMove(20, depthRng) >= 20, "level 20 reaches 20");
assert(M.chooseDepthForMove(8, depthRng) >= 8, "level 8 no longer stuck at 7");
assert(M.chooseDepthForMove(25, depthRng) === 20, "level 25 clamps to 20");

var winBoard = boardFromAscii([
  "bbbbbbb",
  "bYRYbbb",
  "YRRYYbb",
  "RYYRRYb",
  "YRRYYRb",
  "RRYRRRY"
]);
/* Red to move can win in col 3 (0-index). */
var redWins = M.immediateWinColumns(winBoard, C.X);
assert(redWins.indexOf(3) >= 0, "immediateWinColumns finds red win in col 4");
var ordered = M.orderMovesForSearch(winBoard, C.X);
assert(ordered.wins.length > 0, "orderMovesForSearch reports wins");
assert(ordered.ordered[0] === ordered.wins[0], "winning move ordered first");
var deep = M.alphabeta(G.cloneBoard(winBoard), 20, -Infinity, Infinity, C.X, C.X, Math.random);
assert(redWins.indexOf(deep.move) >= 0, "depth-20 alphabeta takes immediate win");

var blockBoard = boardFromAscii(YELLOW_DEFENDS_ROWS);
var blockOrder = M.orderMovesForSearch(blockBoard, C.O);
assert(blockOrder.ordered.length > 0, "defender has ordered moves");
var yellowBlocks = blockingColumns(blockBoard, C.O);
if (blockOrder.wins.length) {
  assert(blockOrder.ordered[0] === blockOrder.wins[0],
    "if defender can win, take win first");
} else {
  assert(yellowBlocks.indexOf(blockOrder.ordered[0]) >= 0,
    "block ordered before quiet moves, first=" + (blockOrder.ordered[0] + 1));
}

console.log("--- double threat (Observe game end) ---");
/* Yellow to move; red already has two immediate wins (cols 2 and 3 UI = 1,2). */
var doubleThreat = boardFromAscii([
  "bbbbbbb",
  "bbbYbbb",
  "bbbYYbb",
  "RRbRObb",
  "YRRYRbb",
  "RRRYYYR"
]);
var redDouble = M.immediateWinColumns(doubleThreat, C.X);
assert(redDouble.length >= 2, "red has a double threat");
assert(redDouble.indexOf(1) >= 0 && redDouble.indexOf(2) >= 0,
  "threats are columns 2 and 3");
assert(M.immediateWinColumns(doubleThreat, C.O).length === 0,
  "yellow has no immediate win");
var lossOrder = M.orderMovesForSearch(doubleThreat, C.O);
assert(lossOrder.forcedLoss, "orderMovesForSearch marks forced loss");
assert(redDouble.indexOf(lossOrder.ordered[0]) >= 0,
  "forced-loss move still blocks one threat");
var lossAb = M.alphabeta(
  G.cloneBoard(doubleThreat), 4, -Infinity, Infinity, C.O, C.O, Math.random
);
assert(lossAb.score === -1000000, "alphabeta scores double threat as loss for yellow");
assert(redDouble.indexOf(lossAb.move) >= 0, "alphabeta still blocks one column");

if (failures.length) {
  console.log("FAIL (" + failures.length + "):");
  var f;
  for (f = 0; f < failures.length; f += 1) {
    console.log("  " + failures[f]);
  }
  process.exit(1);
}

console.log("OK: minimax blocks immediate win for both red and yellow");
