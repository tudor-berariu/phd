#!/usr/local/bin/julia

#=

    In this scenario a couple of agents need to learn how to deliver
    gold from a mine to a deposit. Both mines and deposits emit radio
    signals so agents know in which direction these objectives are.
    Agents get an exponentialy higher reward if they drop the gold at
    the deposit at the same time.
    Will they learn to synchronize?

    Reward = min(2^No. of. dropping agents, 256)

=#

typealias CellInfo Uint8;
typealias RadioSignal Uint8;
typealias AgentState Array{CellInfo, 1};
typealias Action Uint8;

#=
    +---+---+---+---+---+---+---+---+
    |   |   |   |   |   |   |   |N/S|
    +---+---+---+---+---+---+---+---+
=#
const NOTHING   = convert(CellInfo, 0);
const SOMETHING = convert(CellInfo, 1);

#=
    +---+---+---+---+---+---+---+---+
    |   |   |   |   | W | B | A | 1 |
    +---+---+---+---+---+---+---+---+
=#
const AGENT    = SOMETHING | convert(CellInfo, 1<<1);
const BUILDING = SOMETHING | convert(CellInfo, 1<<2);
const WALL     = SOMETHING | convert(CellInfo, 1<<3);

#=
    Agent info
    +---+---+---+---+---+---+---+---+
    |   |   | E | L | 0 | 0 | 1 | 1 |
    +---+---+---+---+---+---+---+---+
=#

const LOADED = SOMETHING | AGENT | convert(CellInfo, 1<<4)
const EMPTY  = SOMETHING | AGENT | convert(CellInfo, 1<<5)

#=
    Building info
    +---+---+---+---+---+---+---+---+
    |   |   | M | W | 0 | 1 | 0 | 1 |
    +---+---+---+---+---+---+---+---+
=#

const WAREHOUSE = SOMETHING | BUILDING | convert(CellInfo, 1<<4);
const MINE      = SOMETHING | BUILDING | convert(CellInfo, 1<<5);

#=
    Actions
=#

const DO_NOTHING  = convert(Action, 0) ;       # 00000000

const UP          = convert(Action, 1<<0);     # 00000001
const DOWN        = convert(Action, 1<<1);     # 00000010
const LEFT        = convert(Action, 1<<2);     # 00000100
const RIGHT       = convert(Action, 1<<3);     # 00001000

const UP_LEFT     = UP   | LEFT ;              # 00000101
const UP_RIGHT    = UP   | RIGHT;              # 00001001
const DOWN_LEFT   = DOWN | LEFT ;              # 00000110
const DOWN_RIGHT  = DOWN | RIGHT;              # 00001010

const DO_LOAD     = convert(Action, 1<<4);     # 00010000
const DO_UNLOAD   = convert(Action, 1<<5);     # 00100000

const validActions = [DO_NOTHING,
                      UP, DOWN, LEFT, RIGHT,
                      UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT,
                      DO_LOAD, DO_UNLOAD];

dy(action::Action) = action & UP > 0 ? -1 : action & DOWN > 0 ? 1 : 0;
dx(action::Action) = action & LEFT > 0 ? -1 : action & RIGHT > 0 ? 1 : 0;

#=
    RadioInfo
    +---+---+---+---+---+---+---+---+
    | MINE SIGNAL   | DEPOSIT SIGNAL|
    +---+---+---+---+---+---+---+---+
=#

const NO_SIGNAL = convert(RadioSignal, 0);
const NORTH     = convert(RadioSignal, 1<<0);
const SOUTH     = convert(RadioSignal, 1<<1);
const EAST      = convert(RadioSignal, 1<<2);
const WEST      = convert(RadioSignal, 1<<3);

const MASK      = convert(RadioSignal, 1 | 1<<1 | 1<<2 | 1<<3);

function newSignal(perceivedSignal::RadioSignal, oldSignal::RadioSignal)
    if perceivedSignal == NO_SIGNAL
        return oldSignal & MASK;
    else
        return (perceivedSignal<<4) | perceivedSignal;
    end
end

#=

=#

const ROW    = 1;
const COLUMN = 2;

sqEuclid(r1, c1, r2, c2) = (r1 - r2) * (r1 - r2) + (c1 - c2) * (c1 - c2);

# ---------------------------------------

const HEIGHT         = 10;
const WIDTH          = 10;
const AGENTS_NO      = 1;
const WAREHOUSES_NO  = 1;
const MINES_NO       = 1;
const RANGE          = 6;
const AGENT_RANGE    = 1;
const MIN_DIST       = 3;
const SEASONS_NO     = 10000;
const EPISODES_NO    = 5000;
# ---------------------------------------

type GlobalState
    board::Array{CellInfo, 2}
    warehouses::Array{Int64, 2}
    mines::Array{Int64, 2}
    agents::Array{Int64, 2}
    wSignal::Array{RadioSignal, 2}
    mSignal::Array{RadioSignal, 2}
end

# ---------------------------------------

function place_buildings()
    warehouses = zeros(Int64, 2, WAREHOUSES_NO);
    mines = zeros(Int64, 2, MINES_NO);

    sqMinDist = MIN_DIST * MIN_DIST;

    notOver = true;
    warehouseIdx = 0;
    mineIdx = 0;

    const MAX_ATTEMPTS = 10;

    while notOver
        while notOver
            attemptNo = 0
            r = 0
            c = 0
            while attemptNo < MAX_ATTEMPTS
                r = rand(MIN_DIST:(1 + HEIGHT - MIN_DIST));
                c = rand(MIN_DIST:(1 + WIDTH - MIN_DIST));
                isGood = true;
                for i in 1:warehouseIdx
                    if sqEuclid(warehouses[ROW, i],
                                warehouses[COLUMN, i], r, c) < sqMinDist
                        attemptNo = attemptNo + 1
                        isGood = false
                        break
                    end
                end
                if isGood
                    for i in 1:mineIdx
                        if sqEuclid(mines[ROW, i],
                                    mines[COLUMN, i], r, c) < sqMinDist
                            attemptNo = attemptNo + 1
                            isGood = false
                            break
                        end
                    end
                end
                if isGood
                    break
                end
                attemptNo = attemptNo + 1
            end
            if attemptNo == MAX_ATTEMPTS
                break
            end
            if warehouseIdx < WAREHOUSES_NO
                warehouseIdx = warehouseIdx + 1
                warehouses[ROW, warehouseIdx] = r
                warehouses[COLUMN, warehouseIdx] = c
            else
                mineIdx = mineIdx + 1
                mines[ROW, mineIdx] = r
                mines[COLUMN, mineIdx] = c
                if mineIdx == MINES_NO
                    notOver = false;
                end
            end
        end
        if notOver
            warehouses = zeros(Int64, 2, WAREHOUSES_NO);
            mines = zeros(Int64, 2, MINES_NO);
            warehouseIdx = 0;
            mineIdx = 0;
        end
    end
    return warehouses, mines;
end

function computeRadioSignal(board, warehouses, mines)
    mSignal = fill(NO_SIGNAL, size(board))
    wSignal = fill(NO_SIGNAL, size(board))
    for r in 1:HEIGHT
        for c in 1:WIDTH
            #-- Warehouses
            bestDistance = Inf
            wIdx = 0
            for w in 1:WAREHOUSES_NO
                d = sqEuclid(warehouses[ROW, w], warehouses[COLUMN, w], r, c);
                if d < bestDistance
                    bestDistance = d;
                    wIdx = w;
                end
            end
            if bestDistance <= RANGE * RANGE
                if r < warehouses[ROW, wIdx]
                    wSignal[r,c] = wSignal[r,c] | SOUTH;
                elseif r > warehouses[ROW, wIdx]
                    wSignal[r,c] = wSignal[r,c] | NORTH;
                end
                if c < warehouses[COLUMN, wIdx]
                    wSignal[r,c] = wSignal[r,c] | EAST;
                elseif c > warehouses[COLUMN, wIdx]
                    wSignal[r,c] = wSignal[r,c] | WEST;
                end
            end
            #-- Mines
            bestDistance = Inf
            mIdx = 0
            for m in 1:MINES_NO
                d = sqEuclid(mines[ROW, m], mines[COLUMN, m], r, c);
                if d < bestDistance
                    bestDistance = d;
                    mIdx = m;
                end
            end
            if bestDistance <= RANGE * RANGE
                if r < mines[ROW, mIdx]
                    mSignal[r,c] = mSignal[r,c] | SOUTH;
                elseif r > mines[ROW, mIdx]
                    mSignal[r,c] = mSignal[r,c] | NORTH;
                end
                if c < mines[COLUMN, mIdx]
                    mSignal[r,c] = mSignal[r,c] | EAST;
                elseif c > mines[COLUMN, mIdx]
                    mSignal[r,c] = mSignal[r,c] | WEST;
                end
            end
        end
    end
    return wSignal, mSignal;
end

function init_scenario()
    board = fill(NOTHING, WIDTH, HEIGHT);
    # Put warehouses and mines
    warehouses, mines = place_buildings();
    for w in 1:WAREHOUSES_NO
        board[warehouses[ROW, w], warehouses[COLUMN, w]] = WAREHOUSE;
    end
    for m in 1:MINES_NO
        board[mines[ROW, m], mines[COLUMN, m]] = MINE;
    end
    # Put agents
    agents = zeros(Int64, 2, AGENTS_NO);
    for a in 1:AGENTS_NO
        agents[ROW, a]    = rand(1:HEIGHT)
        agents[COLUMN, a] = rand(1:WIDTH)
        while (board[agents[ROW, a], agents[COLUMN, a]] != NOTHING)
            agents[ROW, a]    = rand(1:HEIGHT)
            agents[COLUMN, a] = rand(1:WIDTH)
        end
        board[agents[ROW, a], agents[COLUMN, a]] = EMPTY;
    end
    # Compute signals
    wSignal, mSignal = computeRadioSignal(board, warehouses, mines);
    # Pack everything
    return GlobalState(board, warehouses, mines, agents, wSignal, mSignal);
end

function perceiveMap(gs::GlobalState, oldState::AgentState, a::Integer)
    t_idx = gs.agents[ROW, a] - AGENT_RANGE;
    b_idx = gs.agents[ROW, a] + AGENT_RANGE;
    l_idx = gs.agents[COLUMN, a] - AGENT_RANGE;
    r_idx = gs.agents[COLUMN, a] + AGENT_RANGE;

    t_rows = max(0,  1 - t_idx);
    b_rows = max(0, b_idx - HEIGHT);
    l_cols = max(0, 1 - l_idx);
    r_cols = max(0, r_idx - WIDTH);

    const diameter = 2 * AGENT_RANGE + 1;
    const mid =  AGENT_RANGE + 1;

    state =  vcat(fill(WALL, t_rows, diameter),
                  hcat(fill(WALL, diameter - t_rows - b_rows, l_cols),
                       gs.board[max(1, t_idx):min(HEIGHT, b_idx),
                                max(1, l_idx):min(WIDTH, r_idx)],
                       fill(WALL, diameter - t_rows - b_rows, r_cols)),
                  fill(WALL, b_rows, diameter));
    wSignal = newSignal(gs.wSignal[gs.agents[ROW, a], gs.agents[COLUMN, a]],
                        oldState[1]);
    mSignal = newSignal(gs.mSignal[gs.agents[ROW, a], gs.agents[COLUMN, a]],
                        oldState[2]);
    return vcat(wSignal, mSignal, reshape(state, diameter*diameter));
end

function orderAgents(gs::GlobalState, actions::Array{Action, 1})
    return shuffle(1:length(actions));
end

function doLoad(gs::GlobalState, ag::Integer)
    row = gs.agents[ROW, ag];
    col = gs.agents[COLUMN, ag];
    if gs.board[row, col] & EMPTY == EMPTY
        if ((row > 1 && gs.board[row-1,col] == MINE) ||
            (row < HEIGHT && gs.board[row+1,col] == MINE) ||
            (col < WIDTH && gs.board[row,col+1] == MINE) ||
            (col > 1 && gs.board[row,col-1] == MINE))
            gs.board[row, col] = LOADED;
        end
    end
    nothing
end

function doUnload(gs::GlobalState, ag::Integer)
    row = gs.agents[ROW, ag];
    col = gs.agents[COLUMN, ag];
    if gs.board[row, col] & LOADED == LOADED
        if ((row > 1 && gs.board[row-1,col] == WAREHOUSE) ||
            (row < HEIGHT && gs.board[row+1,col] == WAREHOUSE) ||
            (col < WIDTH && gs.board[row,col+1] == WAREHOUSE) ||
            (col > 1 && gs.board[row,col-1] == WAREHOUSE))
            gs.board[row, col] = EMPTY;
            for w in 1:WAREHOUSES_NO
                if sqEuclid(gs.warehouses[ROW,w], gs.warehouses[COLUMN,w],
                            row, col) == 1
                    return w;
                end
            end
        end
    end
    return 0;
end

function doMove(gs::GlobalState, ag::Integer, action::Action)
    row = gs.agents[ROW, ag];
    col = gs.agents[COLUMN, ag];
    if action == DO_NOTHING
        return
    end
    δrows = dy(action)
    δcols = dx(action)
    new_row = min(HEIGHT, max(1, row + δrows))
    new_col = min(WIDTH, max(1, col + δcols))
    if gs.board[new_row, new_col] == NOTHING
        gs.board[new_row, new_col] = gs.board[row, col];
        gs.board[row, col] = NOTHING;
        gs.agents[ROW, ag] = new_row;
        gs.agents[COLUMN, ag] = new_col;
    end
    nothing
end

function getBest(Qs::Dict{AgentState, Dict{Action, Float64}}, s::AgentState)
    AsQs = get!(Qs, s, Dict{Action, Float64}());
    return reduce((best, k) -> AsQs[k] > best[1] ? tuple(AsQs[k], k) : best,
                  tuple(0.0, validActions[rand(1:end)]),
                  keys(AsQs));
end

function chooseAction(s::AgentState,
                      Qs::Dict{AgentState, Dict{Action, Float64}},
                      ϵ::Float64 = 0.01)
    return rand() < ϵ ? validActions[rand(1:end)] : getBest(Qs, s)[2];
end

function log_decay(min_val, max_val, step, steps_no)
    return max_val - log(1+(10-1)*step/steps_no) / log(10) * (max_val-min_val);
end

function update_qs(Qs::Dict{AgentState, Dict{Action, Float64}},
                   s1::AgentState, a::Action, r::Float64, s2::AgentState,
                   γ::Float64, α::Float64)
    AsQs = get!(Qs, s1, Dict{Action, Float64}());
    oldQ = get!(AsQs, a, 0.0);
    AsQs[a] = (1-α) * oldQ + α * (r + γ * getBest(Qs, s2)[1]);
    nothing
end

function learn()
    Qs = Array(Dict{AgentState, Dict{Action, Float64}}, AGENTS_NO);
    for ag in 1:AGENTS_NO
        Qs[ag] = Dict{AgentState, Dict{Action, Float64}}();
    end
    actions = Array(Action, AGENTS_NO);
    prevStates = Array(AgentState, AGENTS_NO);
    for ag in 1:AGENTS_NO
        prevStates[ag] = vcat(NO_SIGNAL, NO_SIGNAL);
    end
    states = Array(AgentState, AGENTS_NO);
    rewards = Array(Float64, AGENTS_NO);
    println("Done")
    # ---
    for season in 1:SEASONS_NO
        println("S $(season)")
        total_score = 0;

        gs = init_scenario()
        for ag in 1:AGENTS_NO
            states[ag] = perceiveMap(gs, prevStates[ag], ag);
        end
        for episode in 1:EPISODES_NO
            #println("E $(episode)")

            # Choose actions
            for ag in 1:AGENTS_NO
                actions[ag] = chooseAction(states[ag], Qs[ag]);
                prevStates[ag] = deepcopy(states[ag]);
            end
            # Do actions
            rewards = zeros(Float64, AGENTS_NO);
            rewarders = zeros(Int64, AGENTS_NO);
            wDrops = zeros(Int64, WAREHOUSES_NO);
            for ag in orderAgents(gs, actions)
                if actions[ag] & DO_LOAD > 0
                    doLoad(gs, ag);
                elseif actions[ag] & DO_UNLOAD > 0
                    wIdx = doUnload(gs, ag);
                    if wIdx > 0
                        wDrops[wIdx] = wDrops[wIdx] + 1;
                    end
                        rewarders[ag] = wIdx;
                else
                    doMove(gs, ag, actions[ag]);
                end
            end
            for ag in 1:AGENTS_NO
                if rewarders[ag] > 0
                    rewards[ag] = 2^wDrops[rewarders[ag]] - 1;
                end
                states[ag] = perceiveMap(gs, prevStates[ag], ag);
                update_qs(Qs[ag], prevStates[ag],
                          actions[ag], rewards[ag], states[ag],
                          0.95, 0.05);
            end
            for w in 1:WAREHOUSES_NO
                total_score = total_score + 2^wDrops[w] - 1;
            end
        end
        println(total_score)
    end
end

const boardChar = Dict{CellInfo, Char}(EMPTY => char('\u25a1'),
                                       LOADED => char('\u25a0'),
                                       WAREHOUSE => char('\u25c7'),
                                       MINE => char('\u25c6'),
                                       NOTHING => ' ',
                                       WALL => 'X');

const signalChar = Dict{RadioSignal, Char}(NO_SIGNAL => char(' '),
                                           WEST => char('\u2190'),
                                           NORTH => char('\u2191'),
                                           EAST => char('\u2192'),
                                           SOUTH => char('\u2193'),
                                           NORTH | WEST => char('\u2196'),
                                           NORTH | EAST => char('\u2197'),
                                           SOUTH | WEST => char('\u2199'),
                                           SOUTH | EAST => char('\u2198'));


function displaySignalMap(signalMap::Array{RadioSignal, 2})
    for row in 1:size(signalMap,1)
        for col in 1:size(signalMap,2)
            print(signalChar[signalMap[row, col]]);
        end
        println();
    end
end

function displayBoard(board)
    for row in 1:size(board,1)
        for col in 1:size(board,2)
            print(boardChar[board[row, col]]);
        end
        println();
    end
end
