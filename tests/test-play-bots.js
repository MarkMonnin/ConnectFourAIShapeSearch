/* Board stacking + full-game smoke test for every play-mode bot (Node). */
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
global.requestAnimationFrame = function (fn) { fn(); return 1; };
global.cancelAnimationFrame = function () {};

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
var PLAY = global.C4_PLAY;

var failures = [];

function fail(msg) {
  failures.push(msg);
}

function assert(cond, msg) {
  if (!cond) {
    fail(msg);
  }
}

function assertColumnStack(board, label) {
  var col;
  var row;
  for (col = 0; col < C.COLS; col += 1) {
    var highest = -1;
    for (row = 0; row < C.ROWS; row += 1) {
      if (G.boardCellValue(board, col, row)) {
        highest = row;
      }
    }
    for (row = 0; row <= highest; row += 1) {
      if (!G.boardCellValue(board, col, row)) {
        fail((label || "board") + ": gap in column " + col + " below row " + row +
          " (highest occupied " + highest + ")");
      }
    }
  }
}

function testDropRowStacksFromBottom() {
  var board = G.emptyBoard();
  var i;
  for (i = 0; i < C.ROWS; i += 1) {
    var row = G.dropRow(board, 3);
    assert(row === i, "dropRow stack: expected row " + i + " got " + row);
    board = G.applyMove(board, i % 2 === 0 ? C.X : C.O, 3);
    assertColumnStack(board, "dropRow stack ply " + i);
  }
  assert(G.dropRow(board, 3) === -1, "full column should return -1");
  assert(G.legalMoves(board).indexOf(3) === -1, "full column not legal");
}

function testVisualRowOrder() {
  var row0Top = PLAY.gameRowToTopPx(0);
  var row1Top = PLAY.gameRowToTopPx(1);
  var row5Top = PLAY.gameRowToTopPx(5);
  assert(row0Top > row1Top, "game row 0 (bottom) should render lower than row 1");
  assert(row1Top > row5Top, "game row 5 (top) should render highest on screen");
  assert(PLAY.visualRow(0) === C.ROWS - 1, "visualRow bottom mapping");
  assert(PLAY.visualRow(C.ROWS - 1) === 0, "visualRow top mapping");
}

function pickLegalMove(app, algoId, board, player) {
  var col = app.pickPlayMove(algoId, board, player, 0);
  var moves = G.legalMoves(board);
  if (typeof col !== "number" || moves.indexOf(col) < 0) {
    return moves[0];
  }
  return col;
}

function playBotGame(app, algoA, algoB, label) {
  var board = G.emptyBoard();
  var current = C.X;
  var plies = 0;
  var maxPlies = C.CELLS + 5;

  while (plies < maxPlies) {
    var winner = G.findWinner(board);
    if (winner || G.isDraw(board)) {
      break;
    }
    var moves = G.legalMoves(board);
    if (!moves.length) {
      break;
    }
    var algo = current === C.X ? algoA : algoB;
    var col = pickLegalMove(app, algo, board, current);
    assert(typeof col === "number", label + ": no move at ply " + plies);
    var row = G.dropRow(board, col);
    assert(row >= 0, label + ": dropRow failed at ply " + plies + " col " + col);
    board = G.applyMove(board, current, col);
    assertColumnStack(board, label + " ply " + plies);
    assert(G.boardCellValue(board, col, row) === current,
      label + ": piece not at expected row " + row + " ply " + plies);
    plies += 1;
    current = G.other(current);
  }

  assert(plies > 0, label + ": game had zero plies");
  assertColumnStack(board, label + " final");
}

function testEveryBot() {
  var app = new C4_APP.Application(true);
  var ids = C.ALL_ALGO_IDS.slice();
  var i;
  for (i = 0; i < ids.length; i += 1) {
    var id = ids[i];
    playBotGame(app, id, "qtable", id + " vs qtable");
    playBotGame(app, "minimax", id, "minimax vs " + id);
  }
}

function testSimulatedObserveColumn() {
  var board = G.emptyBoard();
  var col = 2;
  var plies = 0;
  while (plies < C.ROWS * 2) {
    board = G.applyMove(board, plies % 2 === 0 ? C.X : C.O, col);
    plies += 1;
  }
  assertColumnStack(board, "simulated column");
  var row;
  var lastTop = Infinity;
  for (row = 0; row < C.ROWS; row += 1) {
    if (G.boardCellValue(board, col, row)) {
      var top = PLAY.gameRowToTopPx(row);
      assert(top < lastTop, "occupied rows in one column should descend on screen bottom-up");
      lastTop = top;
    }
  }
}

testDropRowStacksFromBottom();
testVisualRowOrder();
testSimulatedObserveColumn();
testEveryBot();

if (failures.length) {
  console.log("FAIL (" + failures.length + "):");
  var f;
  for (f = 0; f < failures.length; f += 1) {
    console.log("  " + failures[f]);
  }
  process.exit(1);
}

console.log("OK: board stacking + " + C.ALL_ALGO_IDS.length + " bots x2 games");
