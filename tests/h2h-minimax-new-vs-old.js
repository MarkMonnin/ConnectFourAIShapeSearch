/**
 * Head-to-head: new Minimax (double-threat short-circuit ON) vs old (OFF).
 * 1 second search per move each. Stop at lead of 10 wins or 100 games.
 */
var path = require("path");
var root = path.join(__dirname, "..");
process.chdir(root);
global.window = global;
if (typeof global.performance === "undefined") {
  global.performance = { now: function () { return Date.now(); } };
}

require(path.join(root, "c4_js/constants.js"));
require(path.join(root, "c4_js/game.js"));
require(path.join(root, "c4_js/minimax.js"));

var G = global.C4_GAME;
var C = global.C4_CONSTANTS;
var M = global.C4_MINIMAX;

var BUDGET_MS = 1000;
var MAX_GAMES = 100;
var LEAD_WINS = 10;

function pickMove(board, player, useNew) {
  M.setDoubleThreatShortCircuit(useNew);
  var deadline = performance.now() + BUDGET_MS;
  var res = M.chooseMoveUntilDeadline(board, player, deadline, Math.random.bind(Math));
  return res.move;
}

function playGame(newIsRed) {
  var board = G.emptyBoard();
  var player = C.X;
  var guard = 0;
  while (!G.findWinner(board) && !G.isDraw(board) && guard < 42) {
    var useNew = player === C.X ? newIsRed : !newIsRed;
    var col = pickMove(board, player, useNew);
    if (typeof col !== "number" || G.dropRow(board, col) < 0) {
      var legal = G.legalMoves(board);
      col = legal.length ? legal[0] : null;
    }
    if (col === null) {
      break;
    }
    board = G.applyMove(board, player, col);
    player = G.other(player);
    guard += 1;
  }
  var winner = G.findWinner(board);
  if (!winner) {
    return "draw";
  }
  var newMark = newIsRed ? C.X : C.O;
  return winner === newMark ? "new" : "old";
}

var newWins = 0;
var oldWins = 0;
var draws = 0;
var games = 0;
var t0 = Date.now();

while (games < MAX_GAMES) {
  var newIsRed = (games % 2) === 0;
  var result = playGame(newIsRed);
  games += 1;
  if (result === "new") {
    newWins += 1;
  } else if (result === "old") {
    oldWins += 1;
  } else {
    draws += 1;
  }
  var lead = Math.abs(newWins - oldWins);
  if (games % 5 === 0 || lead >= LEAD_WINS) {
    console.log(
      "progress games=" + games +
      " new=" + newWins + " old=" + oldWins + " draws=" + draws +
      " lead=" + lead
    );
  }
  if (lead >= LEAD_WINS) {
    break;
  }
}

M.setDoubleThreatShortCircuit(true);

var elapsedSec = ((Date.now() - t0) / 1000).toFixed(1);
var winner;
if (newWins > oldWins) {
  winner = "NEW";
} else if (oldWins > newWins) {
  winner = "OLD";
} else {
  winner = "TIE";
}

console.log("H2H Minimax new vs old");
console.log("budget_ms_per_move=" + BUDGET_MS);
console.log("stop=first lead of " + LEAD_WINS + " wins OR best of " + MAX_GAMES);
console.log("games=" + games + " newWins=" + newWins + " oldWins=" + oldWins +
  " draws=" + draws);
console.log("winner=" + winner);
console.log("elapsed_s=" + elapsedSec);
console.log("colors=alternating (game0 new=red)");
