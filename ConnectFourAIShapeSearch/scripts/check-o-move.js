const path = require("path");
const root = path.join(__dirname, "..");
process.chdir(root);
global.window = global;
global.performance = { now: function () { return Date.now(); } };
require(path.join(root, "c4_js/constants.js"));
require(path.join(root, "c4_js/game.js"));
require(path.join(root, "c4_js/minimax.js"));

const G = global.C4_GAME;
const C = global.C4_CONSTANTS;
const M = global.C4_MINIMAX;

const xMoves = [2, 2, 3, 4, 7, 5, 3, 1, 1, 2, 3].map(function (c) { return c - 1; });
const oMoves = [4, 6, 4, 5, 4, 5, 5, 1, 4, 2].map(function (c) { return c - 1; });

function allRootScores(board, player, depth) {
  var moves = M.orderMovesForSearch(board, player).ordered;
  var out = [];
  var i;
  for (i = 0; i < moves.length; i += 1) {
    var b = G.cloneBoard(board);
    var token = G.makeMove(b, player, moves[i]);
    var res = M.alphabeta(
      b, depth - 1, -Infinity, Infinity, G.other(player), player,
      function () { return 0.5; }
    );
    G.undoMove(b, token);
    out.push({ col: moves[i] + 1, score: res.score });
  }
  out.sort(function (a, b) { return b.score - a.score; });
  return out;
}

/* After X plays second 2 (ply 3), O played 6 in the game. */
var board = G.emptyBoard();
board = G.applyMove(board, C.X, xMoves[0]);
board = G.applyMove(board, C.O, oMoves[0]);
board = G.applyMove(board, C.X, xMoves[1]);
console.log("Position where game O played 6:");
console.log("d6 scores", JSON.stringify(allRootScores(board, C.O, 6)));
console.log("d8 scores", JSON.stringify(allRootScores(board, C.O, 8)));

/* Many 1s searches with Math.random */
var hits = {};
var i;
for (i = 0; i < 30; i += 1) {
  var res = M.chooseMoveUntilDeadline(
    G.cloneBoard(board), C.O, Date.now() + 1000, Math.random.bind(Math)
  );
  var k = String(res.move + 1);
  hits[k] = (hits[k] || 0) + 1;
}
console.log("30x 1s picks", JSON.stringify(hits));
