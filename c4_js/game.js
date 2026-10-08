(function (global) {

  "use strict";



  var C = global.C4_CONSTANTS;

  var ROWS = C.ROWS;

  var COLS = C.COLS;

  var X = C.X;

  var O = C.O;

  var ZERO = 0n;

  var FULL_MASK = ZERO;
  var COL_TOP_BITS = [];



  function other(player) {

    return player === X ? O : X;

  }



  function cellBit(col, row) {

    return 1n << BigInt(col + row * COLS);

  }

  (function initBitboardConstants() {
    var col;
    var row;
    for (col = 0; col < COLS; col += 1) {
      COL_TOP_BITS[col] = cellBit(col, ROWS - 1);
    }
    for (row = 0; row < ROWS; row += 1) {
      for (col = 0; col < COLS; col += 1) {
        FULL_MASK |= cellBit(col, row);
      }
    }
  }());



  function emptyBoard() {

    return { red: ZERO, yellow: ZERO };

  }



  function cloneBoard(board) {

    return { red: board.red, yellow: board.yellow };

  }



  function mirrorColumn(col) {

    return COLS - 1 - col;

  }



  function mirrorBits(bits) {

    var out = ZERO;

    var col;

    var row;

    for (row = 0; row < ROWS; row += 1) {

      for (col = 0; col < COLS; col += 1) {

        if ((bits & cellBit(col, row)) !== ZERO) {

          out |= cellBit(mirrorColumn(col), row);

        }

      }

    }

    return out;

  }



  function mirrorBoard(board) {

    return { red: mirrorBits(board.red), yellow: mirrorBits(board.yellow) };

  }



  /**
   * Double a training trajectory with left-right mirrors.
   * Policy steps need { board, player, action, input?, moves? }.
   * Value steps need { board }.
   */
  function expandTrajectoryWithMirrors(trajectory) {

    var out = [];

    var i;

    for (i = 0; i < trajectory.length; i += 1) {

      var step = trajectory[i];

      out.push(step);

      if (!step || !step.board) {

        continue;

      }

      var mb = mirrorBoard(step.board);

      if (typeof step.action === "number" && step.player) {

        var mirrored = {

          board: mb,

          player: step.player,

          action: mirrorColumn(step.action),

          input: boardToInput(mb, step.player)

        };

        if (step.moves) {

          mirrored.moves = legalMoves(mb);

        }

        out.push(mirrored);

      } else {

        out.push({ board: mb });

      }

    }

    return out;

  }



  function mask(board) {

    return board.red | board.yellow;

  }



  function boardKey(board) {

    return board.red.toString() + ":" + board.yellow.toString();

  }



  function playerBits(board, player) {

    return player === X ? board.red : board.yellow;

  }



  function dropRow(board, col) {

    var occupied = mask(board);

    var row;

    for (row = 0; row < ROWS; row += 1) {

      if ((occupied & cellBit(col, row)) === ZERO) {

        return row;

      }

    }

    return -1;

  }



  function legalMoves(board) {

    var occupied = mask(board);

    var moves = [];

    var col;

    for (col = 0; col < COLS; col += 1) {

      if ((occupied & COL_TOP_BITS[col]) === ZERO) {

        moves.push(col);

      }

    }

    return moves;

  }



  /* Search uses makeMove/undoMove in place (one clone at search entry). Depth-10 empty-board
     benchmark (tests/benchmark-minimax-make-undo.js): make/undo ~516ms median vs applyMove
     ~20550ms (~40x). applyMove clones every ply; keep it for UI/replay, not tree search. */

  function makeMove(board, player, col) {

    var row = dropRow(board, col);

    if (row < 0) {

      return null;

    }

    var bit = cellBit(col, row);

    if (player === X) {

      board.red |= bit;

    } else {

      board.yellow |= bit;

    }

    return { player: player, col: col, row: row, bit: bit };

  }



  function undoMove(board, token) {

    if (!token) {

      return;

    }

    if (token.player === X) {

      board.red &= ~token.bit;

    } else {

      board.yellow &= ~token.bit;

    }

  }



  function applyMove(board, player, col) {

    var next = cloneBoard(board);

    if (makeMove(next, player, col) === null) {

      return board;

    }

    return next;

  }



  function isBoardFull(board) {

    return mask(board) === FULL_MASK;

  }



  /* Start-bit masks so shift-based win checks cannot wrap across row edges.
     Layout is col + row*COLS; raw >>1 / >>6 / >>8 false-positive without these. */
  var HORIZ_WIN_START = ZERO;
  var DIAG6_WIN_START = ZERO;
  var DIAG8_WIN_START = ZERO;
  (function initWinStartMasks() {
    var r;
    var c;
    for (r = 0; r < ROWS; r += 1) {
      for (c = 0; c < COLS - 3; c += 1) {
        HORIZ_WIN_START |= cellBit(c, r);
      }
    }
    for (r = 0; r < ROWS - 3; r += 1) {
      for (c = 3; c < COLS; c += 1) {
        DIAG6_WIN_START |= cellBit(c, r);
      }
      for (c = 0; c < COLS - 3; c += 1) {
        DIAG8_WIN_START |= cellBit(c, r);
      }
    }
  }());

  function hasWin(bits) {
    if (bits === ZERO) {
      return false;
    }
    var m;
    /* vertical */
    m = bits & (bits >> 7n);
    if ((m & (m >> 14n)) !== ZERO) {
      return true;
    }
    /* horizontal */
    m = bits & (bits >> 1n);
    m = m & (m >> 2n);
    if ((m & HORIZ_WIN_START) !== ZERO) {
      return true;
    }
    /* diagonal up-right */
    m = bits & (bits >> 8n);
    m = m & (m >> 16n);
    if ((m & DIAG8_WIN_START) !== ZERO) {
      return true;
    }
    /* diagonal up-left */
    m = bits & (bits >> 6n);
    m = m & (m >> 12n);
    if ((m & DIAG6_WIN_START) !== ZERO) {
      return true;
    }
    return false;
  }



  function findWinner(board) {

    if (hasWin(board.red)) {

      return X;

    }

    if (hasWin(board.yellow)) {

      return O;

    }

    return null;

  }



  function isDraw(board) {

    return !findWinner(board) && isBoardFull(board);

  }



  function resultForPlayer(winner, player) {

    if (!winner) {

      return "draw";

    }

    return winner === player ? "win" : "loss";

  }



  function randomMove(board) {

    var moves = legalMoves(board);

    return moves[Math.floor(Math.random() * moves.length)];

  }



  function buildAnalysisWindows() {

    var wins = [];

    var r;

    var c;

    for (r = 0; r < ROWS; r += 1) {

      for (c = 0; c < COLS - 3; c += 1) {

        wins.push([

          cellBit(c, r),

          cellBit(c + 1, r),

          cellBit(c + 2, r),

          cellBit(c + 3, r)

        ]);

      }

    }

    for (c = 0; c < COLS; c += 1) {

      for (r = 0; r < ROWS - 3; r += 1) {

        wins.push([

          cellBit(c, r),

          cellBit(c, r + 1),

          cellBit(c, r + 2),

          cellBit(c, r + 3)

        ]);

      }

    }

    for (c = 0; c < COLS - 3; c += 1) {

      for (r = 0; r < ROWS - 3; r += 1) {

        wins.push([

          cellBit(c, r),

          cellBit(c + 1, r + 1),

          cellBit(c + 2, r + 2),

          cellBit(c + 3, r + 3)

        ]);

      }

    }

    for (c = 0; c < COLS - 3; c += 1) {

      for (r = 3; r < ROWS; r += 1) {

        wins.push([

          cellBit(c, r),

          cellBit(c + 1, r - 1),

          cellBit(c + 2, r - 2),

          cellBit(c + 3, r - 3)

        ]);

      }

    }

    return wins;

  }



  var ANALYSIS_WINDOWS = buildAnalysisWindows();

  var BOARD_FEATURE_DIM = C.BOARD_FEATURE_DIM || 35;



  function countWindowThreats(red, yellow, player) {

    var my3 = 0;

    var opp3 = 0;

    var my2 = 0;

    var opp2 = 0;

    var w;

    var i;

    for (w = 0; w < ANALYSIS_WINDOWS.length; w += 1) {

      var windowBits = ANALYSIS_WINDOWS[w];

      var p = 0;

      var o = 0;

      var e = 0;

      var bit;

      for (i = 0; i < 4; i += 1) {

        bit = windowBits[i];

        if (red & bit) {

          if (player === X) {

            p += 1;

          } else {

            o += 1;

          }

        } else if (yellow & bit) {

          if (player === O) {

            p += 1;

          } else {

            o += 1;

          }

        } else {

          e += 1;

        }

      }

      if (o > 0 && p > 0) {

        continue;

      }

      if (p === 3 && e === 1) {

        my3 += 1;

      } else if (o === 3 && e === 1) {

        opp3 += 1;

      } else if (p === 2 && e === 2) {

        my2 += 1;

      } else if (o === 2 && e === 2) {

        opp2 += 1;

      }

    }

    return { my3: my3, opp3: opp3, my2: my2, opp2: opp2 };

  }



  function centerColumnCount(bits) {

    var total = 0;

    var r;

    for (r = 0; r < ROWS; r += 1) {

      if (bits & cellBit(3, r)) {

        total += 1;

      }

    }

    return total;

  }



  function winningColumnsFor(board, player) {

    var cols = [];

    var moves = legalMoves(board);

    var scratch = cloneBoard(board);

    var i;

    for (i = 0; i < moves.length; i += 1) {

      var col = moves[i];

      var token = makeMove(scratch, player, col);

      if (token === null) {

        continue;

      }

      if (hasWin(player === X ? scratch.red : scratch.yellow)) {

        cols.push(col);

      }

      undoMove(scratch, token);

    }

    return cols;

  }



  function boardAnalysisFeatures(board, player) {

    var features = [];

    var col;

    var threats = countWindowThreats(board.red, board.yellow, player);

    var winCols = winningColumnsFor(board, player);

    var blockCols = winningColumnsFor(board, other(player));

    var moves = legalMoves(board);

    for (col = 0; col < COLS; col += 1) {

      var row = dropRow(board, col);

      features.push(row < 0 ? 1 : (row + 1) / ROWS);

    }

    for (col = 0; col < COLS; col += 1) {

      features.push(winCols.indexOf(col) >= 0 ? 1 : 0);

    }

    for (col = 0; col < COLS; col += 1) {

      features.push(blockCols.indexOf(col) >= 0 ? 1 : 0);

    }

    for (col = 0; col < COLS; col += 1) {

      features.push(dropRow(board, col) < 0 ? 1 : 0);

    }

    features.push(threats.my3 / 10);

    features.push(threats.opp3 / 10);

    features.push(threats.my2 / 15);

    features.push(threats.opp2 / 15);

    features.push(centerColumnCount(player === X ? board.red : board.yellow) / ROWS);

    features.push(centerColumnCount(player === X ? board.yellow : board.red) / ROWS);

    features.push(moves.length / COLS);

    return features;

  }



  function boardCellValue(board, col, row) {

    var bit = cellBit(col, row);

    if (board.red & bit) {

      return X;

    }

    if (board.yellow & bit) {

      return O;

    }

    return null;

  }



  function boardToRedInput(board) {

    var values = [];

    var row;

    var col;

    for (row = 0; row < ROWS; row += 1) {

      for (col = 0; col < COLS; col += 1) {

        var cell = boardCellValue(board, col, row);

        if (cell === X) {

          values.push(1);

        } else if (cell === O) {

          values.push(-1);

        } else {

          values.push(0);

        }

      }

    }

    var feat = boardAnalysisFeatures(board, X);

    var f;

    for (f = 0; f < feat.length; f += 1) {

      values.push(feat[f]);

    }

    return values;

  }



  function boardToInput(board, player) {

    var input = [];

    var row;

    var col;

    for (row = 0; row < ROWS; row += 1) {

      for (col = 0; col < COLS; col += 1) {

        var bit = cellBit(col, row);

        if ((mask(board) & bit) === ZERO) {

          input.push(0);

        } else if (playerBits(board, player) & bit) {

          input.push(1);

        } else {

          input.push(-1);

        }

      }

    }

    input.push(player === X ? 1 : -1);

    var feat = boardAnalysisFeatures(board, player);

    var f;

    for (f = 0; f < feat.length; f += 1) {

      input.push(feat[f]);

    }

    return input;

  }



  global.C4_GAME = {

    other: other,

    emptyBoard: emptyBoard,

    cloneBoard: cloneBoard,

    mirrorColumn: mirrorColumn,

    mirrorBoard: mirrorBoard,

    expandTrajectoryWithMirrors: expandTrajectoryWithMirrors,

    boardKey: boardKey,

    mask: mask,

    cellBit: cellBit,

    playerBits: playerBits,

    legalMoves: legalMoves,

    dropRow: dropRow,

    applyMove: applyMove,

    makeMove: makeMove,

    undoMove: undoMove,

    isBoardFull: isBoardFull,

    findWinner: findWinner,

    hasWin: hasWin,

    isDraw: isDraw,

    resultForPlayer: resultForPlayer,

    randomMove: randomMove,

    boardToInput: boardToInput,

    boardToRedInput: boardToRedInput,

    boardAnalysisFeatures: boardAnalysisFeatures,

    BOARD_FEATURE_DIM: BOARD_FEATURE_DIM,

    POLICY_INPUT_DIM: C.POLICY_INPUT_DIM || (ROWS * COLS + 1 + BOARD_FEATURE_DIM),

    VALUE_INPUT_DIM: C.VALUE_INPUT_DIM || (ROWS * COLS + BOARD_FEATURE_DIM),

    boardCellValue: boardCellValue

  };

})(window);


