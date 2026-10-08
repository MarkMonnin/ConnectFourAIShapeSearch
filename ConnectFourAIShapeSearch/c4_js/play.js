(function (global) {
  "use strict";

  var G = global.C4_GAME;
  var C = global.C4_CONSTANTS;
  var APP = global.C4_APP;

  var CELL = 48;
  var GAP = 8;
  var PITCH = CELL + GAP;
  var FRAME_PAD = 8;
  var LABEL_H = 28;
  var GRID_H = C.ROWS * CELL + (C.ROWS - 1) * GAP;
  var GRID_TOP = LABEL_H + FRAME_PAD;
  var DROP_MS = 1000;

  var SEARCH_SEC_OPTIONS = [0.5, 1, 2, 3, 5, 10, 15, 30];

  function playerNumberForMark(mark) {
    return mark === C.X ? 1 : 2;
  }

  function markForPlayerNumber(n) {
    return n === 1 ? C.X : C.O;
  }

  function discClass(mark) {
    return mark === C.X ? "play-disc-red" : "play-disc-yellow";
  }

  function visualRow(gameRow) {
    return C.ROWS - 1 - gameRow;
  }

  function buildAlgoOptions(selectEl, catalog) {
    var i;
    selectEl.innerHTML = "";
    var humanOpt = document.createElement("option");
    humanOpt.value = APP.PLAY_HUMAN;
    humanOpt.textContent = "Human";
    selectEl.appendChild(humanOpt);
    if (catalog && catalog.length) {
      for (i = 0; i < catalog.length; i += 1) {
        var opt = document.createElement("option");
        opt.value = catalog[i].id;
        opt.textContent = catalog[i].name || catalog[i].label || catalog[i].id;
        selectEl.appendChild(opt);
      }
      return;
    }
    var ids = C.ALL_ALGO_IDS.slice();
    for (i = 0; i < ids.length; i += 1) {
      opt = document.createElement("option");
      opt.value = ids[i];
      opt.textContent = APP.displayName(ids[i]);
      selectEl.appendChild(opt);
    }
  }

  function buildSearchOptions(selectEl) {
    var i;
    selectEl.innerHTML = "";
    for (i = 0; i < SEARCH_SEC_OPTIONS.length; i += 1) {
      var opt = document.createElement("option");
      opt.value = String(SEARCH_SEC_OPTIONS[i]);
      opt.textContent = SEARCH_SEC_OPTIONS[i] + " sec";
      if (SEARCH_SEC_OPTIONS[i] === 1) {
        opt.selected = true;
      }
      selectEl.appendChild(opt);
    }
  }

  function PlayMode(app, rootEl) {
    this.app = app;
    this.rootEl = rootEl;
    this.board = G.emptyBoard();
    this.currentMark = C.X;
    this.gameActive = false;
    this.animating = false;
    this.config = null;
    this.els = {};
    this.buildChrome();
    this.refreshCatalog();
    this.syncSearchVisibility();
  }

  PlayMode.prototype.refreshCatalog = function () {
    var catalog = this.app.getPlayCatalog ? this.app.getPlayCatalog() : null;
    if (this.els.p1Kind) {
      var v1 = this.els.p1Kind.value;
      var v2 = this.els.p2Kind.value;
      buildAlgoOptions(this.els.p1Kind, catalog);
      buildAlgoOptions(this.els.p2Kind, catalog);
      if (v1) {
        this.els.p1Kind.value = v1;
      }
      if (v2) {
        this.els.p2Kind.value = v2;
      }
    }
  };

  PlayMode.prototype.buildChrome = function () {
    var self = this;
    this.rootEl.innerHTML =
      "<div class=\"play-panel\">" +
        "<p class=\"hint play-hint\">Player 1 is red and moves first. Columns are numbered 1-7 left to right.</p>" +
        "<div class=\"play-setup\">" +
          "<div class=\"play-player-config\">" +
            "<label class=\"play-label\">Player 1 (red)</label>" +
            "<select id=\"playP1Kind\" class=\"play-select\"></select>" +
            "<select id=\"playP1Search\" class=\"play-select play-search-select\" hidden></select>" +
          "</div>" +
          "<div class=\"play-player-config\">" +
            "<label class=\"play-label\">Player 2 (yellow)</label>" +
            "<select id=\"playP2Kind\" class=\"play-select\"></select>" +
            "<select id=\"playP2Search\" class=\"play-select play-search-select\" hidden></select>" +
          "</div>" +
          "<div class=\"play-actions\">" +
            "<button type=\"button\" id=\"btnPlayStart\" class=\"btn btn-primary\">Start game</button>" +
            "<button type=\"button\" id=\"btnPlayNew\" class=\"btn\" disabled>New game</button>" +
          "</div>" +
        "</div>" +
        "<div id=\"playStatus\" class=\"play-status\">Choose players, then start a game.</div>" +
        "<div id=\"playBoardHost\" class=\"play-board-host\"></div>" +
        "<p class=\"play-log-caption\">Move log (think time is search only; each drop animates ~1s)</p>" +
        "<div id=\"playMoveLog\" class=\"play-move-log\" aria-live=\"polite\"></div>" +
      "</div>";

    this.els.p1Kind = this.rootEl.querySelector("#playP1Kind");
    this.els.p2Kind = this.rootEl.querySelector("#playP2Kind");
    this.els.p1Search = this.rootEl.querySelector("#playP1Search");
    this.els.p2Search = this.rootEl.querySelector("#playP2Search");
    this.els.start = this.rootEl.querySelector("#btnPlayStart");
    this.els.newGame = this.rootEl.querySelector("#btnPlayNew");
    this.els.status = this.rootEl.querySelector("#playStatus");
    this.els.boardHost = this.rootEl.querySelector("#playBoardHost");
    this.els.moveLog = this.rootEl.querySelector("#playMoveLog");
    this.logLines = [];

    buildSearchOptions(this.els.p1Search);
    buildSearchOptions(this.els.p2Search);

    this.els.p1Kind.addEventListener("change", function () {
      self.syncSearchVisibility();
    });
    this.els.p2Kind.addEventListener("change", function () {
      self.syncSearchVisibility();
    });
    this.els.start.addEventListener("click", function () {
      self.startGame();
    });
    this.els.newGame.addEventListener("click", function () {
      self.resetToSetup();
    });

    this.buildBoardDom();
    this.renderBoard();
  };

  PlayMode.prototype.kindHasTimedSearch = function (kind) {
    if (kind === APP.PLAY_HUMAN || kind === "random") {
      return false;
    }
    if (APP.playAlgoHasTimedSearch(kind)) {
      return true;
    }
    /* Learned Shape Search bots: allow a think-time budget. */
    return !!this.app.getPlayCatalog;
  };

  PlayMode.prototype.syncSearchVisibility = function () {
    this.els.p1Search.hidden = !this.kindHasTimedSearch(this.els.p1Kind.value);
    this.els.p2Search.hidden = !this.kindHasTimedSearch(this.els.p2Kind.value);
  };

  PlayMode.prototype.readSideConfig = function (kindEl, searchEl) {
    var kind = kindEl.value;
    if (kind === APP.PLAY_HUMAN) {
      return { type: "human", label: null, algoId: null, searchSec: 0 };
    }
    var searchSec = parseFloat(searchEl.value) || 1;
    var label = kindEl.options[kindEl.selectedIndex] ?
      kindEl.options[kindEl.selectedIndex].textContent : kind;
    return {
      type: "bot",
      algoId: kind,
      label: label,
      searchSec: this.kindHasTimedSearch(kind) ? searchSec : 0
    };
  };

  PlayMode.prototype.readConfig = function () {
    return {
      p1: this.readSideConfig(this.els.p1Kind, this.els.p1Search),
      p2: this.readSideConfig(this.els.p2Kind, this.els.p2Search)
    };
  };

  PlayMode.prototype.sideForMark = function (mark) {
    if (!this.config) {
      return null;
    }
    return mark === C.X ? this.config.p1 : this.config.p2;
  };

  PlayMode.prototype.sideLabel = function (side, playerNum) {
    if (side.type === "human") {
      return "Player " + playerNum;
    }
    if (side.searchSec > 0) {
      return side.label + " (" + side.searchSec + " sec)";
    }
    return side.label;
  };

  PlayMode.prototype.setStatus = function (text, kind) {
    this.els.status.textContent = text;
    this.els.status.className = "play-status" + (kind ? " play-status-" + kind : "");
  };

  PlayMode.prototype.logActorLabel = function (side, num) {
    if (side.type === "human") {
      return "Human";
    }
    return this.sideLabel(side, num);
  };

  PlayMode.prototype.clearMoveLog = function () {
    this.logLines = [];
    if (this.els.moveLog) {
      this.els.moveLog.innerHTML = "";
    }
  };

  PlayMode.prototype.renderMoveLog = function () {
    var html = "";
    var i;
    for (i = 0; i < this.logLines.length; i += 1) {
      var line = this.logLines[i];
      html += "<div class=\"play-log-line play-log-" + line.kind + "\">" + line.text + "</div>";
    }
    this.els.moveLog.innerHTML = html;
    this.els.moveLog.scrollTop = this.els.moveLog.scrollHeight;
  };

  PlayMode.prototype.appendLog = function (text, kind) {
    this.logLines.push({ text: text, kind: kind || "played" });
    this.renderMoveLog();
  };

  PlayMode.prototype.logBotThink = function (label, elapsedSec, depth) {
    var text = label + " thought for " + elapsedSec.toFixed(1) + " seconds";
    if (typeof depth === "number" && depth > 0) {
      text += " (depth " + depth + ")";
    }
    this.appendLog(text, "thought");
  };

  PlayMode.prototype.botSearchDepth = function (algoId) {
    if (this.app.getLastPlayDepth) {
      var labDepth = this.app.getLastPlayDepth(algoId);
      if (typeof labDepth === "number" && labDepth > 0) {
        return labDepth;
      }
    }
    if (this.app.registry && this.app.registry[algoId] &&
        typeof this.app.registry[algoId].lastDepth === "number") {
      return this.app.registry[algoId].lastDepth || 0;
    }
    return 0;
  };

  PlayMode.prototype.setSetupLocked = function (locked) {
    this.els.p1Kind.disabled = locked;
    this.els.p2Kind.disabled = locked;
    this.els.p1Search.disabled = locked;
    this.els.p2Search.disabled = locked;
    this.els.start.disabled = locked;
    this.els.newGame.disabled = !locked;
  };

  PlayMode.prototype.buildBoardDom = function () {
    var col;
    var row;
    var boardW = FRAME_PAD * 2 + C.COLS * CELL + (C.COLS - 1) * GAP;
    var boardH = LABEL_H + FRAME_PAD + GRID_H + FRAME_PAD;
    var html =
      "<div class=\"play-board-outer\" style=\"width:" + boardW + "px;height:" + boardH + "px\">" +
      "<div class=\"play-col-labels\">";
    for (col = 0; col < C.COLS; col += 1) {
      html += "<span class=\"play-col-num\">" + (col + 1) + "</span>";
    }
    var gridW = C.COLS * CELL + (C.COLS - 1) * GAP;
    html +=
      "</div>" +
      "<div class=\"play-disc-layer\" style=\"top:" + GRID_TOP + "px;left:" + FRAME_PAD +
      "px;width:" + gridW + "px;height:" + GRID_H + "px\"></div>" +
      "<div class=\"play-board-face\"><div class=\"play-grid\">";
    for (row = C.ROWS - 1; row >= 0; row -= 1) {
      for (col = 0; col < C.COLS; col += 1) {
        html += "<div class=\"play-cell\" data-col=\"" + col + "\" data-row=\"" + row + "\"></div>";
      }
    }
    html += "</div></div><div class=\"play-columns\" style=\"top:" +
      GRID_TOP + "px;height:" + GRID_H + "px\">";
    for (col = 0; col < C.COLS; col += 1) {
      html +=
        "<button type=\"button\" class=\"play-col-btn\" data-col=\"" + col + "\" aria-label=\"Column " +
        (col + 1) + "\"></button>";
    }
    html += "</div></div>";
    this.els.boardHost.innerHTML = html;
    this.els.discLayer = this.els.boardHost.querySelector(".play-disc-layer");
    this.els.colBtns = this.els.boardHost.querySelectorAll(".play-col-btn");
    var self = this;
    var i;
    for (i = 0; i < this.els.colBtns.length; i += 1) {
      this.els.colBtns[i].addEventListener("click", function () {
        var col = parseInt(this.getAttribute("data-col"), 10);
        self.onColumnClick(col);
      });
    }
  };

  PlayMode.prototype.discLeft = function (col) {
    return col * PITCH;
  };

  PlayMode.prototype.discTop = function (row) {
    return visualRow(row) * PITCH;
  };

  PlayMode.prototype.renderBoard = function () {
    var layer = this.els.discLayer;
    layer.innerHTML = "";
    var row;
    var col;
    for (row = 0; row < C.ROWS; row += 1) {
      for (col = 0; col < C.COLS; col += 1) {
        var mark = G.boardCellValue(this.board, col, row);
        if (mark) {
          this.placeDiscElement(col, row, mark, false);
        }
      }
    }
  };

  PlayMode.prototype.placeDiscElement = function (col, row, mark, animateFromTop, onDropComplete) {
    var disc = document.createElement("div");
    disc.className = "play-disc " + discClass(mark);
    disc.style.left = this.discLeft(col) + "px";
    var targetTop = this.discTop(row);
    if (animateFromTop) {
      disc.style.top = (-PITCH) + "px";
      disc.style.transition = "none";
    } else {
      disc.style.top = targetTop + "px";
    }
    this.els.discLayer.appendChild(disc);
    if (animateFromTop) {
      disc.getBoundingClientRect();
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          disc.style.transition = "top " + DROP_MS + "ms cubic-bezier(0.33, 1, 0.68, 1)";
          disc.style.top = targetTop + "px";
        });
      });
      if (onDropComplete) {
        setTimeout(onDropComplete, DROP_MS + 50);
      }
    }
    return disc;
  };

  PlayMode.prototype.setColumnsEnabled = function (enabled) {
    var i;
    for (i = 0; i < this.els.colBtns.length; i += 1) {
      this.els.colBtns[i].disabled = !enabled;
    }
    this.els.boardHost.classList.toggle("play-human-turn", enabled);
  };

  PlayMode.prototype.resetToSetup = function () {
    this.gameActive = false;
    this.animating = false;
    this.config = null;
    this.board = G.emptyBoard();
    this.currentMark = C.X;
    this.setSetupLocked(false);
    this.syncSearchVisibility();
    this.setColumnsEnabled(false);
    this.renderBoard();
    this.clearMoveLog();
    this.setStatus("Choose players, then start a game.");
  };

  PlayMode.prototype.startGame = function () {
    this.config = this.readConfig();
    this.board = G.emptyBoard();
    this.currentMark = C.X;
    this.gameActive = true;
    this.animating = false;
    this.clearMoveLog();
    this.setSetupLocked(true);
    this.renderBoard();
    this.setColumnsEnabled(false);
    this.setStatus("Game started. " + this.turnIntroText());
    var self = this;
    setTimeout(function () {
      self.runTurn();
    }, 300);
  };

  PlayMode.prototype.turnIntroText = function () {
    var side = this.sideForMark(this.currentMark);
    var num = playerNumberForMark(this.currentMark);
    if (side.type === "human") {
      return "Player " + num + "'s turn.";
    }
    if (side.searchSec > 0) {
      return this.sideLabel(side, num) + " will move first.";
    }
    return side.label + " will move first.";
  };

  PlayMode.prototype.onColumnClick = function (col) {
    if (!this.gameActive || this.animating) {
      return;
    }
    var side = this.sideForMark(this.currentMark);
    if (!side || side.type !== "human") {
      return;
    }
    if (G.dropRow(this.board, col) < 0) {
      this.setStatus("That column is full. Pick another.", "warn");
      return;
    }
    this.commitMove(col, side);
  };

  PlayMode.prototype.runTurn = function () {
    if (!this.gameActive || this.animating) {
      return;
    }
    var side = this.sideForMark(this.currentMark);
    var num = playerNumberForMark(this.currentMark);
    if (side.type === "human") {
      this.setStatus("Player " + num + "'s turn");
      this.setColumnsEnabled(true);
      return;
    }
    this.setColumnsEnabled(false);
    var self = this;
    var label = this.logActorLabel(side, num);
    if (side.searchSec > 0) {
      this.setStatus(label + " is thinking");
    } else {
      this.setStatus(side.label + " is thinking");
    }
    setTimeout(function () {
      if (!self.gameActive) {
        return;
      }
      var t0 = performance.now();
      var col = self.app.pickPlayMove(
        side.algoId, self.board, self.currentMark, side.searchSec
      );
      var elapsedSec = (performance.now() - t0) / 1000;
      if (typeof col !== "number" || G.dropRow(self.board, col) < 0) {
        var moves = G.legalMoves(self.board);
        col = moves.length ? moves[0] : col;
      }
      var thinkDepth = side.searchSec > 0 ? self.botSearchDepth(side.algoId) : 0;
      if (side.searchSec > 0) {
        self.logBotThink(label, elapsedSec, thinkDepth);
        if (thinkDepth > 0) {
          self.setStatus(label + " thought to depth " + thinkDepth);
        }
      }
      var played = label + " played column " + (col + 1);
      if (thinkDepth > 0) {
        played += " (depth " + thinkDepth + ")";
      }
      self.appendLog(played, "played");
      self.commitMove(col, side);
    }, 0);
  };

  PlayMode.prototype.commitMove = function (col, side) {
    var self = this;
    var mark = this.currentMark;
    var num = playerNumberForMark(mark);
    var row = G.dropRow(this.board, col);
    if (row < 0) {
      this.setStatus("No legal move in that column.", "warn");
      if (side.type === "human") {
        this.setColumnsEnabled(true);
      } else {
        this.runTurn();
      }
      return;
    }
    if (side.type === "human") {
      this.appendLog("Human played column " + (col + 1), "played");
    }
    /* Apply before the drop animation so a missed/cancelled callback cannot
       leave the log/status ahead of the board (and so endgame always paints). */
    this.board = G.applyMove(this.board, mark, col);
    this.animating = true;
    this.setColumnsEnabled(false);
    var msg;
    if (side.type === "human") {
      msg = "Player " + num + " plays in column " + (col + 1);
    } else if (side.searchSec > 0) {
      msg = this.sideLabel(side, num) + " plays in column " + (col + 1);
    } else {
      msg = side.label + " plays in column " + (col + 1);
    }
    this.setStatus(msg, "move");
    var finished = false;
    var disc = this.placeDiscElement(col, row, mark, true, function () {
      if (finished) {
        return;
      }
      finished = true;
      if (disc.parentNode) {
        disc.parentNode.removeChild(disc);
      }
      self.renderBoard();
      self.animating = false;
      self.checkEndOrContinue();
    });
  };

  PlayMode.prototype.checkEndOrContinue = function () {
    var winner = G.findWinner(this.board);
    if (winner) {
      this.finishGame(winner);
      return;
    }
    if (G.isDraw(this.board)) {
      this.finishGame(null);
      return;
    }
    this.currentMark = G.other(this.currentMark);
    this.runTurn();
  };

  PlayMode.prototype.finishGame = function (winner) {
    this.gameActive = false;
    this.setColumnsEnabled(false);
    this.renderBoard();
    if (winner) {
      var num = playerNumberForMark(winner);
      var side = this.sideForMark(winner);
      if (side.type === "human") {
        this.setStatus("Player " + num + " wins!", "win");
      } else if (side.searchSec > 0) {
        this.setStatus(this.sideLabel(side, num) + " wins!", "win");
      } else {
        this.setStatus(side.label + " wins!", "win");
      }
      this.appendLog(this.logActorLabel(side, num) + " wins!", "result");
    } else {
      this.setStatus("Draw game.", "draw");
      this.appendLog("Draw game.", "result");
    }
    this.setSetupLocked(false);
    this.els.newGame.disabled = false;
  };

  PlayMode.prototype.onTabHidden = function () {
    if (!this.gameActive && !this.animating) {
      return;
    }
    this.gameActive = false;
    this.animating = false;
    this.setColumnsEnabled(false);
    this.setSetupLocked(false);
    this.setStatus("Game stopped. Switch back anytime to start a new game.");
  };

  function initTabs(app, tabConfig) {
    tabConfig = tabConfig || {};
    var lab = tabConfig.lab || null;
    var tabSearch = tabConfig.tabSearch || document.getElementById("tabSearch");
    var tabTrain = tabConfig.tabTrain || document.getElementById("tabTrain");
    var tabPlay = tabConfig.tabPlay || document.getElementById("tabPlay");
    var panelSearch = tabConfig.panelSearch || document.getElementById("panelSearch");
    var panelTrain = tabConfig.panelTrain || document.getElementById("panelTrain");
    var panelPlay = tabConfig.panelPlay || document.getElementById("panelPlay");
    var playMode = new PlayMode(app, panelPlay);

    function setTabActive(btn, on) {
      if (!btn) {
        return;
      }
      btn.classList.toggle("tab-active", on);
      btn.setAttribute("aria-selected", on ? "true" : "false");
    }

    function activateTab(name) {
      var isSearch = name === "search";
      var isTrain = name === "train";
      var isPlay = name === "play";
      setTabActive(tabSearch, isSearch);
      setTabActive(tabTrain, isTrain);
      setTabActive(tabPlay, isPlay);
      if (panelSearch) {
        panelSearch.hidden = !isSearch;
      }
      if (panelTrain) {
        panelTrain.hidden = !isTrain;
      }
      if (panelPlay) {
        panelPlay.hidden = !isPlay;
      }
      if (lab) {
        lab.setActiveTab(isSearch ? "search" : (isTrain ? "train" : "observe"));
      }
      if (isPlay) {
        playMode.refreshCatalog();
      } else {
        playMode.onTabHidden();
      }
    }

    if (tabSearch) {
      tabSearch.addEventListener("click", function () {
        activateTab("search");
      });
    }
    if (tabTrain) {
      tabTrain.addEventListener("click", function () {
        activateTab("train");
      });
    }
    if (tabPlay) {
      tabPlay.addEventListener("click", function () {
        activateTab("play");
      });
    }

    activateTab("search");
    return playMode;
  }

  global.C4_PLAY = {
    initTabs: initTabs,
    PlayMode: PlayMode,
    visualRow: visualRow,
    gameRowToTopPx: function (gameRow) {
      return GRID_TOP + visualRow(gameRow) * PITCH;
    }
  };
})(window);
