const API = 'http://localhost:5050/api';

/* ─── utilities ─────────────────────────────────────────────────────────── */
function log(id, msg, cls='') {
  const el = document.getElementById(id);
  const d = document.createElement('div');
  d.className = 'log-entry ' + cls;
  d.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
}
function setBtn(id, html, disabled=false) {
  const b = document.getElementById(id);
  b.innerHTML = html; b.disabled = disabled;
}
function stat(id, v) { document.getElementById(id).textContent = v; }

/* ─── tab ───────────────────────────────────────────────────────────────── */
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i)=>
    t.classList.toggle('active', ['sudoku','2048'][i]===name));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('panel-'+name).classList.add('active');
}

/* ══════════════════════════════ SUDOKU ════════════════════════════════════ */
let sSteps=[], sTimer=null, sIdx=0;
let sPlaced=new Set(), sOrigPuzzle=null;

function buildSudokuGrid() {
  const board = document.getElementById('sudoku-board');
  board.innerHTML = '';
  for (let i=0; i<81; i++) {
    const cell = document.createElement('div');
    cell.className = 'sudoku-cell empty';
    cell.dataset.idx = i;
    cell.dataset.row = Math.floor(i/9);
    cell.dataset.col = i%9;
    board.appendChild(cell);
  }
}

function renderSudoku(boardArr, highlightIdx=-1) {
  const cells = document.querySelectorAll('.sudoku-cell');
  const fixed = sOrigPuzzle || Array(81).fill(0);
  cells.forEach((cell, i) => {
    const v = boardArr[i];
    cell.textContent = v > 0 ? v : '';
    cell.className = 'sudoku-cell';
    if (v === 0) {
      cell.classList.add('empty');
    } else if (fixed[i] !== 0) {
      cell.classList.add('fixed');
    } else if (i === highlightIdx) {
      cell.classList.add('highlighted');
    } else {
      cell.classList.add('placed');
    }
  });
}

function onDiffChange() {
  if (sTimer) { clearInterval(sTimer); sTimer = null; }
  sSteps = []; sIdx = 0; sPlaced = new Set(); sOrigPuzzle = null;
  buildSudokuGrid();
  stat('s-step','0'); stat('s-total','—'); stat('s-filled','0'); stat('s-reward','—');
  document.getElementById('sudoku-prog').style.width = '0%';
  document.getElementById('sudoku-prog-txt').textContent = 'Ready';
  setBtn('btn-s-solve','▶ Solve Step-by-Step', true);
}

async function trainSudoku() {
  if (sTimer) { clearInterval(sTimer); sTimer = null; }
  const diff = document.getElementById('sudoku-diff').value;
  setBtn('btn-s-train',
    '<span class="training-indicator"><span class="pulse-dot"></span>Training…</span>', true);
  log('s-log', `Training Q-Learning agent — ${diff}…`);

  try {
    await fetch(`${API}/sudoku/train`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({difficulty: diff})
    });
  } catch(e) {
    log('s-log','Cannot reach backend. Is app.py running?','err');
    setBtn('btn-s-train','⚡ Train RL Agent', false); return;
  }

  // Poll until ready
  const poll = setInterval(async () => {
    try {
      const r = await fetch(`${API}/status`);
      const d = await r.json();
      if (d.training.sudoku[diff] === 'ready') {
        clearInterval(poll);
        setBtn('btn-s-train','⚡ Re-Train', false);
        setBtn('btn-s-solve','▶ Solve Step-by-Step', false);
        log('s-log','Agent ready!','ok');

        // Show initial puzzle
        const pr = await fetch(`${API}/sudoku/puzzle?difficulty=${diff}`);
        const pd = await pr.json();
        sOrigPuzzle = pd.board;
        renderSudoku(pd.board);
      }
    } catch(_) {}
  }, 1200);
}

async function solveSudoku() {
  if (sTimer) { clearInterval(sTimer); sTimer = null; }
  const diff = document.getElementById('sudoku-diff').value;
  setBtn('btn-s-solve','⏳ Computing…', true);
  log('s-log','Requesting RL solution…');

  let data;
  try {
    const r = await fetch(`${API}/sudoku/solve`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({difficulty: diff, board: sOrigPuzzle})
    });
    data = await r.json();
  } catch(e) {
    log('s-log','Error: '+e.message,'err');
    setBtn('btn-s-solve','▶ Solve Step-by-Step', false); return;
  }

  if (!data.steps || data.steps.length === 0) {
    log('s-log','No steps returned.','err');
    setBtn('btn-s-solve','▶ Solve Step-by-Step', false); return;
  }

  sSteps = data.steps;
  sIdx = 0;
  sOrigPuzzle = data.puzzle;
  // Reset board to initial puzzle
  renderSudoku(data.puzzle);

  stat('s-total', data.total_steps);
  log('s-log', `Solved=${data.solved}  Total steps: ${data.total_steps}`, data.solved?'ok':'err');
  setBtn('btn-s-solve','▶ Solve Step-by-Step', false);

  const speed = () => parseInt(document.getElementById('s-speed').value);

  sTimer = setInterval(() => {
    if (sIdx >= sSteps.length) {
      clearInterval(sTimer); sTimer = null;
      return;
    }
    const step = sSteps[sIdx];
    const hlIdx = step.action ? step.action.idx : -1;
    renderSudoku(step.board, hlIdx);

    const filled = step.board.filter(v=>v>0).length;
    const total  = 81;
    document.getElementById('sudoku-prog').style.width =
      (filled / total * 100).toFixed(1) + '%';
    document.getElementById('sudoku-prog-txt').textContent = step.message || '';

    stat('s-step', sIdx + 1);
    stat('s-filled', filled);
    if (step.reward !== undefined) stat('s-reward', step.reward);

    if (step.message && (step.message.includes('SOLVED') || sIdx % 5 === 0))
      log('s-log', step.message, step.message.includes('SOLVED')?'ok':'');

    sIdx++;
  }, speed());
}

function resetSudoku() {
  if (sTimer) { clearInterval(sTimer); sTimer = null; }
  sSteps=[]; sIdx=0; sPlaced=new Set();
  if (sOrigPuzzle) renderSudoku(sOrigPuzzle);
  else buildSudokuGrid();
  stat('s-step','0'); stat('s-total','—');
  stat('s-filled', sOrigPuzzle ? sOrigPuzzle.filter(v=>v>0).length : 0);
  stat('s-reward','—');
  document.getElementById('sudoku-prog').style.width='0%';
  document.getElementById('sudoku-prog-txt').textContent='Reset';
  log('s-log','Board reset to initial puzzle.');
}

/* ══════════════════════════════ 2048 ══════════════════════════════════════ */
let gSteps=[], gTimer=null, gIdx=0;

function tileClass(v) {
  if (v === 0) return 'tile-0';
  if (v > 2048) return 'tile-high';
  return 'tile-' + v;
}

function buildBoard2048() {
  const el = document.getElementById('board-2048');
  el.innerHTML = '';
  for (let i=0; i<16; i++) {
    const t = document.createElement('div');
    t.className = 'tile-2048 tile-0';
    el.appendChild(t);
  }
}

function renderBoard2048(board2d) {
  const tiles = document.querySelectorAll('.tile-2048');
  tiles.forEach((t,i) => {
    const r=Math.floor(i/4), c=i%4;
    const v = board2d[r][c];
    t.className = 'tile-2048 ' + tileClass(v);
    t.textContent = v > 0 ? v : '';
  });
}

function highlightDir(dir) {
  ['UP','DOWN','LEFT','RIGHT'].forEach(d => {
    const el = document.getElementById('dir-'+d);
    if (el) el.classList.toggle('active', d===dir);
  });
}

async function train2048() {
  setBtn('btn-g-train',
    '<span class="training-indicator"><span class="pulse-dot"></span>Training 1000 eps…</span>', true);
  log('g-log','Training Q-Learning agent (1000 episodes)…');

  try {
    await fetch(`${API}/2048/train`, {
      method:'POST', headers:{'Content-Type':'application/json'}
    });
  } catch(e) {
    log('g-log','Cannot reach backend.','err');
    setBtn('btn-g-train','⚡ Train RL Agent', false); return;
  }

  const poll = setInterval(async () => {
    try {
      const r = await fetch(`${API}/status`);
      const d = await r.json();
      if (d.training['2048'] === 'ready') {
        clearInterval(poll);
        setBtn('btn-g-train','⚡ Re-Train', false);
        setBtn('btn-g-play','▶ Play Episode', false);
        log('g-log','Agent trained and ready!','ok');
      }
    } catch(_) {}
  }, 1500);
}

async function play2048() {
  if (gTimer) { clearInterval(gTimer); gTimer = null; }
  setBtn('btn-g-play','⏳ Computing…', true);
  log('g-log','Running trained agent episode…');

  let data;
  try {
    const r = await fetch(`${API}/2048/solve`, {
      method:'POST', headers:{'Content-Type':'application/json'}
    });
    data = await r.json();
  } catch(e) {
    log('g-log','Error: '+e.message,'err');
    setBtn('btn-g-play','▶ Play Episode', false); return;
  }

  gSteps = data.steps;
  gIdx = 0;
  log('g-log', `Episode ready: ${data.total_steps} moves`, 'hi');
  setBtn('btn-g-play','▶ Play Episode', false);

  const speed = () => parseInt(document.getElementById('g-speed').value);

  gTimer = setInterval(() => {
    if (gIdx >= gSteps.length) {
      clearInterval(gTimer); gTimer = null;
      return;
    }
    const step = gSteps[gIdx];
    renderBoard2048(step.board);
    if (step.action_name) highlightDir(step.action_name);
    else highlightDir(null);

    const pct = (gIdx / Math.max(gSteps.length-1,1) * 100).toFixed(1);
    document.getElementById('g-prog').style.width = pct + '%';
    document.getElementById('g-prog-txt').textContent = step.message || '';

    stat('g-step',  gIdx);
    stat('g-score', step.score || 0);
    const maxTile = Math.max(...step.board.flat());
    stat('g-max', maxTile > 0 ? maxTile : '—');
    if (step.reward !== undefined) stat('g-reward', step.reward);

    if (gIdx % 15 === 0 || step.message.includes('Game Over'))
      log('g-log', step.message, step.message.includes('Game Over')?'ok':'');

    gIdx++;
  }, speed());
}

function reset2048() {
  if (gTimer) { clearInterval(gTimer); gTimer = null; }
  gSteps=[]; gIdx=0;
  buildBoard2048();
  stat('g-step','0'); stat('g-score','0'); stat('g-max','—'); stat('g-reward','—');
  document.getElementById('g-prog').style.width='0%';
  document.getElementById('g-prog-txt').textContent='Reset';
  highlightDir(null);
  log('g-log','Board reset.');
}

/* ─── init ───────────────────────────────────────────────────────────────── */
buildSudokuGrid();
buildBoard2048();