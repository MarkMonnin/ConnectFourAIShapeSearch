(function (global) {

  "use strict";

  var G = global.C4_GAME;
  var C = global.C4_CONSTANTS;

  var WIN_SCORE = 1000000;
  var CENTER_COL = 3;
  var CENTER_BONUS = 6;
  var CENTER_MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6];
  /* FractionalMinimax level depth cap (was 7; raised so R8+ eval is not stuck at d7). */
  var MAX_FRACTIONAL_DEPTH = 20;
  /* Timed search: check the clock every N nodes (perf.now is relatively expensive). */
  var TIME_CHECK_MASK = 31;
  /** When true, 2+ opponent immediate wins short-circuit as a forced loss. */
  var DOUBLE_THREAT_SHORT_CIRCUIT = true;

  function setDoubleThreatShortCircuit(enabled) {
    DOUBLE_THREAT_SHORT_CIRCUIT = !!enabled;
  }

  function getDoubleThreatShortCircuit() {
    return DOUBLE_THREAT_SHORT_CIRCUIT;
  }

  function makeSeededRng(seed) {
    var state = seed >>> 0;
    return function () {
      state = (state + 0x6D2B79F5) >>> 0;
      var t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function buildWindowBitIndices() {
    var wins = [];
    var r;
    var c;
    var i;
    for (r = 0; r < C.ROWS; r += 1) {
      for (c = 0; c < C.COLS - 3; c += 1) {
        wins.push([
          G.cellBit(c, r),
          G.cellBit(c + 1, r),
          G.cellBit(c + 2, r),
          G.cellBit(c + 3, r)
        ]);
      }
    }
    for (c = 0; c < C.COLS; c += 1) {
      for (r = 0; r < C.ROWS - 3; r += 1) {
        wins.push([
          G.cellBit(c, r),
          G.cellBit(c, r + 1),
          G.cellBit(c, r + 2),
          G.cellBit(c, r + 3)
        ]);
      }
    }
    for (c = 0; c < C.COLS - 3; c += 1) {
      for (r = 0; r < C.ROWS - 3; r += 1) {
        wins.push([
          G.cellBit(c, r),
          G.cellBit(c + 1, r + 1),
          G.cellBit(c + 2, r + 2),
          G.cellBit(c + 3, r + 3)
        ]);
      }
    }
    for (c = 0; c < C.COLS - 3; c += 1) {
      for (r = 3; r < C.ROWS; r += 1) {
        wins.push([
          G.cellBit(c, r),
          G.cellBit(c + 1, r - 1),
          G.cellBit(c + 2, r - 2),
          G.cellBit(c + 3, r - 3)
        ]);
      }
    }
    return wins;
  }

  var WINDOW_BITS = buildWindowBitIndices();

  var CENTER_BITS = [];
  for (var cr = 0; cr < C.ROWS; cr += 1) {
    CENTER_BITS.push(G.cellBit(CENTER_COL, cr));
  }

  function scoreWindowBits(red, yellow, windowBits, player) {
    var p = 0;
    var o = 0;
    var e = 0;
    var i;
    var bit;
    for (i = 0; i < 4; i += 1) {
      bit = windowBits[i];
      if (red & bit) {
        if (player === C.X) {
          p += 1;
        } else {
          o += 1;
        }
      } else if (yellow & bit) {
        if (player === C.O) {
          p += 1;
        } else {
          o += 1;
        }
      } else {
        e += 1;
      }
    }
    if (o > 0 && p > 0) {
      return 0;
    }
    if (p === 4) {
      return 10000;
    }
    if (p === 3 && e === 1) {
      return 100;
    }
    if (p === 2 && e === 2) {
      return 10;
    }
    if (p === 1 && e === 3) {
      return 1;
    }
    if (o === 3 && e === 1) {
      return -120;
    }
    if (o === 2 && e === 2) {
      return -15;
    }
    return 0;
  }

  function centerColumnScoreBits(red, yellow, player) {
    var total = 0;
    var i;
    var playerBits = player === C.X ? red : yellow;
    for (i = 0; i < CENTER_BITS.length; i += 1) {
      if (playerBits & CENTER_BITS[i]) {
        total += CENTER_BONUS;
      }
    }
    return total;
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

  function evaluate(board, player) {
    var terminal = terminalScore(board, player);
    if (terminal !== null) {
      return terminal;
    }
    var total = 0;
    var w;
    for (w = 0; w < WINDOW_BITS.length; w += 1) {
      total += scoreWindowBits(board.red, board.yellow, WINDOW_BITS[w], player);
    }
    total += centerColumnScoreBits(board.red, board.yellow, player);
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

  /** Columns where player wins on the next drop (in-place probe on search board). */
  function immediateWinColumns(board, player) {
    var moves = G.legalMoves(board);
    var wins = [];
    var i;
    var token;
    for (i = 0; i < moves.length; i += 1) {
      token = G.makeMove(board, player, moves[i]);
      if (!token) {
        continue;
      }
      if (G.hasWin(player === C.X ? board.red : board.yellow)) {
        wins.push(moves[i]);
      }
      G.undoMove(board, token);
    }
    return wins;
  }

  /**
   * Threat-aware order: own wins, then blocks, then center-out.
   * forcedLoss: opponent has two-or-more immediate wins (double threat); this
   * side cannot stop both.
   */
  function orderMovesForSearch(board, sideToMove) {
    var moves = G.legalMoves(board);
    if (moves.length <= 1) {
      return { wins: moves.slice(), ordered: moves.slice(), forcedLoss: false };
    }
    var wins = immediateWinColumns(board, sideToMove);
    if (wins.length) {
      return { wins: wins, ordered: wins.slice(), forcedLoss: false };
    }
    var blocks = immediateWinColumns(board, G.other(sideToMove));
    if (DOUBLE_THREAT_SHORT_CIRCUIT && blocks.length >= 2) {
      return { wins: [], ordered: blocks.slice(), forcedLoss: true };
    }
    var ordered = [];
    var seen = {};
    var i;
    var j;
    var col;
    for (i = 0; i < blocks.length; i += 1) {
      col = blocks[i];
      if (!seen[col]) {
        seen[col] = true;
        ordered.push(col);
      }
    }
    for (i = 0; i < CENTER_MOVE_ORDER.length; i += 1) {
      col = CENTER_MOVE_ORDER[i];
      if (seen[col]) {
        continue;
      }
      for (j = 0; j < moves.length; j += 1) {
        if (moves[j] === col) {
          seen[col] = true;
          ordered.push(col);
          break;
        }
      }
    }
    return {
      wins: [],
      ordered: ordered.length ? ordered : moves.slice(),
      forcedLoss: false
    };
  }

  /* In-place make/undo only; see makeMove comment in game.js for depth-10 benchmark (~40x vs applyMove).
     Tried transposition table + cross-game outcome cache (10k proven win/loss entries); TT gave no
     clear depth gain at 5s; outcome cache searched deeper but lost 8-1 vs plain minimax in 10-game
     series at 1s/move. Removed - not worth the complexity.
     Depth-20 help: immediate-win short-circuit + win/block move ordering (big alpha-beta prune). */

  function alphabeta(board, depth, alpha, beta, sideToMove, evalPlayer, rng, timeCtx) {
    if (timeCtx) {
      timeCtx.nodes = (timeCtx.nodes || 0) + 1;
      if ((timeCtx.nodes & TIME_CHECK_MASK) === 0 &&
          performance.now() >= timeCtx.deadline) {
        timeCtx.aborted = true;
        return { score: evaluate(board, evalPlayer), move: null };
      }
    }
    var maximizing = sideToMove === evalPlayer;
    var terminal = terminalScore(board, evalPlayer);
    if (terminal !== null || depth === 0) {
      return { score: terminal !== null ? terminal : evaluate(board, evalPlayer), move: null };
    }

    var orderedInfo = orderMovesForSearch(board, sideToMove);
    if (orderedInfo.wins.length) {
      /* Forced win this ply - no need to search further. */
      return {
        score: maximizing ? WIN_SCORE : -WIN_SCORE,
        move: orderedInfo.wins[0]
      };
    }
    if (orderedInfo.forcedLoss) {
      /* Opponent has two immediate winning columns; one move cannot block both. */
      return {
        score: maximizing ? -WIN_SCORE : WIN_SCORE,
        move: orderedInfo.ordered[0]
      };
    }
    var moves = orderedInfo.ordered;
    if (!moves.length) {
      return { score: evaluate(board, evalPlayer), move: null };
    }

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
        var sc = alphabeta(board, depth - 1, alpha, beta, G.other(sideToMove), evalPlayer, rng, timeCtx).score;
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
      if (timeCtx && timeCtx.aborted) {
        break;
      }
      token = G.makeMove(board, sideToMove, moves[i]);
      if (!token) {
        continue;
      }
      sc = alphabeta(board, depth - 1, alpha, beta, G.other(sideToMove), evalPlayer, rng, timeCtx).score;
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

  function searchBoardFrom(board) {
    return G.cloneBoard(board);
  }

  function chooseMoveUntilDeadline(board, player, deadline, rng) {
    var moves = G.legalMoves(board);
    if (!moves.length) {
      return { move: null, depth: 0 };
    }
    if (moves.length === 1) {
      return { move: moves[0], depth: 0 };
    }
    rng = rng || Math.random.bind(Math);
    var searchBoard = searchBoardFrom(board);
    var bestMove = moves[0];
    var depth = 1;
    var lastDepth = 0;
    var maxDepth = C.CELLS;
    var timeCtx = { deadline: deadline, aborted: false, nodes: 0 };
    while (depth <= maxDepth && performance.now() < deadline) {
      timeCtx.aborted = false;
      var res = alphabeta(searchBoard, depth, -Infinity, Infinity, player, player, rng, timeCtx);
      if (timeCtx.aborted) {
        break;
      }
      if (res.move !== null) {
        bestMove = res.move;
      }
      lastDepth = depth;
      depth += 1;
    }
    return { move: bestMove, depth: lastDepth };
  }

  function MinimaxOpponent(depth, rng) {
    this.depth = Math.max(1, depth);
    this.rng = rng || Math.random.bind(Math);
  }

  MinimaxOpponent.prototype.chooseMove = function (board, player) {
    var searchBoard = searchBoardFrom(board);
    var res = alphabeta(searchBoard, this.depth, -Infinity, Infinity, player, player, this.rng);
    return res.move !== null ? res.move : G.legalMoves(board)[0];
  };

  function normalizeLevel(level) {
    var n = Math.round(Number(level) * 10) / 10;
    if (!isFinite(n)) {
      return 0.1;
    }
    /* toFixed clears IEEE noise so "0.30000000000000004" never leaks into UI. */
    return Number(n.toFixed(1));
  }

  function upgradeChance(level) {
    var normalized = normalizeLevel(level);
    return (Math.round(normalized * 10) % 10) / 10;
  }

  function chooseDepthForMove(level, rng) {
    var normalized = normalizeLevel(level);
    if (normalized < 1) {
      return 1;
    }
    var base = Math.floor(normalized);
    var depth = rng() < upgradeChance(normalized) ? base + 1 : base;
    return Math.min(Math.max(depth, 1), MAX_FRACTIONAL_DEPTH);
  }

  function FractionalMinimaxOpponent(level, rng) {
    this.level = normalizeLevel(level);
    this.rng = rng || Math.random.bind(Math);
    this._cache = {};
  }

  FractionalMinimaxOpponent.prototype._opp = function (depth) {
    if (!this._cache[depth]) {
      this._cache[depth] = new MinimaxOpponent(depth, this.rng);
    }
    return this._cache[depth];
  };

  FractionalMinimaxOpponent.prototype.chooseMove = function (board, player) {
    if (this.level < 1) {
      if (this.rng() < this.level) {
        return this._opp(1).chooseMove(board, player);
      }
      var moves = G.legalMoves(board);
      return moves[Math.floor(this.rng() * moves.length)];
    }
    return this._opp(chooseDepthForMove(this.level, this.rng)).chooseMove(board, player);
  };

  global.C4_MINIMAX = {
    normalizeLevel: normalizeLevel,
    FractionalMinimaxOpponent: FractionalMinimaxOpponent,
    MinimaxOpponent: MinimaxOpponent,
    alphabeta: alphabeta,
    evaluate: evaluate,
    chooseMoveUntilDeadline: chooseMoveUntilDeadline,
    chooseDepthForMove: chooseDepthForMove,
    orderMovesCenterOut: orderMovesCenterOut,
    orderMovesForSearch: orderMovesForSearch,
    immediateWinColumns: immediateWinColumns,
    setDoubleThreatShortCircuit: setDoubleThreatShortCircuit,
    getDoubleThreatShortCircuit: getDoubleThreatShortCircuit,
    MAX_FRACTIONAL_DEPTH: MAX_FRACTIONAL_DEPTH,
    makeSeededRng: makeSeededRng
  };
})(window);
