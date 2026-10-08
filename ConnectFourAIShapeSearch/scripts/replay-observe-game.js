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

function boardStr(board) {
  var s = "";
  var r;
  var c;
  var red = G.playerBits(board, C.X);
  var yel = G.playerBits(board, C.O);
  for (r = C.ROWS - 1; r >= 0; r -= 1) {
    var row = "";
    for (c = 0; c < C.COLS; c += 1) {
      var bit = G.cellBit(c, r);
      if ((red & bit) !== 0n) {
        row += "X";
      } else if ((yel & bit) !== 0n) {
        row += "O";
      } else {
        row += ".";
      }
    }
    s += row + "\n";
  }
  return s;
}

function threatCount(board, player) {
  var n = 0;
  var moves = G.legalMoves(board);
  var i;
  for (i = 0; i < moves.length; i += 1) {
    if (G.findWinner(G.applyMove(board, player, moves[i])) === player) {
      n += 1;
    }
  }
  return n;
}

function scoreAtDepth(board, player, depth) {
  var opp = new M.MinimaxOpponent(depth, function () { return 0.5; });
  var searchBoard = G.cloneBoard(board);
  var res = M.alphabeta(
    searchBoard, depth, -Infinity, Infinity, player, player, function () { return 0.5; }
  );
  return { move: res.move, score: res.score };
}

var board = G.emptyBoard();
var xi = 0;
var oi = 0;
var ply = 0;

while (xi < xMoves.length) {
  ply += 1;
  board = G.applyMove(board, C.X, xMoves[xi]);
  console.log(
    "Ply", ply, "X played", xMoves[xi] + 1,
    "winner", G.findWinner(board),
    "Xthreats", threatCount(board, C.X),
    "Othreats", threatCount(board, C.O)
  );
  xi += 1;
  if (G.findWinner(board)) {
    break;
  }
  if (oi >= oMoves.length) {
    break;
  }

  var before = G.cloneBoard(board);
  var d6 = scoreAtDepth(before, C.O, 6);
  var d8 = scoreAtDepth(before, C.O, 8);
  var timed = M.chooseMoveUntilDeadline(
    G.cloneBoard(before),
    C.O,
    Date.now() + 1000,
    function () { return 0.5; }
  );
  var played = oMoves[oi];
  console.log(
    "  O to move; played", played + 1,
    "| d6", d6.move + 1, "sc", d6.score,
    "| d8", d8.move + 1, "sc", d8.score,
    "| 1s", timed.move + 1, "d", timed.depth,
    "| Xthreats", threatCount(before, C.X)
  );
  if (threatCount(before, C.X) >= 2) {
    console.log("  DOUBLE THREAT already present before O move:");
    console.log(boardStr(before));
  }
  ply += 1;
  board = G.applyMove(board, C.O, played);
  console.log(
    "Ply", ply, "O played", played + 1,
    "winner", G.findWinner(board),
    "Xthreats", threatCount(board, C.X)
  );
  oi += 1;
  if (G.findWinner(board)) {
    break;
  }
}
