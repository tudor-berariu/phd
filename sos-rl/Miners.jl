# Tudor Berariu, 2015

module Miners

#using Base.Test;

export GlobalState, AgentState, Action, AgentState
export AGENTS_NO, WAREHOUSES_NO
export initialState, perceiveMap, validActions, doActions

# -----------------------------------------------------------------------------

#= Global constants =#

const HEIGHT           = 15;
const WIDTH            = 15;
const AGENTS_NO        = 15;
const WAREHOUSES_NO    = 3;
const MINES_NO         = 3;
const RANGE            = 10;
const AGENT_RANGE      = 1.5;
const MIN_DIST         = 4;

# -----------------------------------------------------------------------------

#= Type aliases =#

typealias CellInfo    Uint8;
typealias Signal      Uint8;
typealias AgentState  Array{Uint8, 1};
typealias Action      Uint8;

# -----------------------------------------------------------------------------

#= CellInfo =#

#=
    +---+---+---+---+---+---+---+---+
    |   |   |   |   |   |   |   |N/S|
    +---+---+---+---+---+---+---+---+
=#
const NOTHING   = convert(CellInfo, 0);
const SOMETHING = convert(CellInfo, 1);

isNothing(cell::CellInfo)   = cell == 0;
isSomething(cell::CellInfo) = cell > 0;

#=
    Entity info
    +---+---+---+---+---+---+---+---+
    |   |   |   |   | W | B | A | 1 |
    +---+---+---+---+---+---+---+---+
=#
const AGENT    = SOMETHING | convert(CellInfo, 1<<1);
const BUILDING = SOMETHING | convert(CellInfo, 1<<2);
const WALL     = SOMETHING | convert(CellInfo, 1<<3);

isAgent(cell::CellInfo)    = (cell & AGENT) == AGENT;
isBuilding(cell::CellInfo) = (cell & BUILDING) == BUILDING;
isWall(cell::CellInfo)     = (cell & WALL) == WALL;

#=
    Agent info
    +---+---+---+---+---+---+---+---+
    |   |   | E | L | 0 | 0 | 1 | 1 |
    +---+---+---+---+---+---+---+---+
=#

const LOADED = SOMETHING | AGENT | convert(CellInfo, 1<<4)
const EMPTY  = SOMETHING | AGENT | convert(CellInfo, 1<<5)

isLoaded(agent::CellInfo) = (agent & LOADED) == LOADED;
isEmpty(agent::CellInfo)  = (agent & EMPTY) == EMPTY;

#=
    Building info
    +---+---+---+---+---+---+---+---+
    |   |   | M | W | 0 | 1 | 0 | 1 |
    +---+---+---+---+---+---+---+---+
=#

const WAREHOUSE = SOMETHING | BUILDING | convert(CellInfo, 1<<4);
const MINE      = SOMETHING | BUILDING | convert(CellInfo, 1<<5);

isWarehouse(building::CellInfo) = (building & WAREHOUSE) == WAREHOUSE;
isMine(building::CellInfo)      = (building & MINE     ) == MINE;

const CellInfoToChar = Dict{CellInfo, Char}(EMPTY     => char('\u26c4'),
                                            LOADED    => char('\u26c7'),
                                            WAREHOUSE => char('\u26ea'),
                                            MINE      => char('\u233e'),
                                            NOTHING   => ' ',
                                            WALL      => char('\u2588'));

# -----------------------------------------------------------------------------

#=
    Actions
=#

const DO_NOTHING     = convert(Action, 0) ;              # 00000000

const DO_MOVE        = convert(Action, 1<<0);            # ----0001
const DO_LOAD        = convert(Action, 1<<1);            # 00000010
const DO_UNLOAD      = convert(Action, 1<<2);            # 00000100
const DO_RANDOM_MOVE = convert(Action, 1<<3);            # 00001000

const UP             = DO_MOVE | convert(Action, 1<<4);  # 00010001
const DOWN           = DO_MOVE | convert(Action, 1<<5);  # 00100001
const LEFT           = DO_MOVE | convert(Action, 1<<6);  # 01000001
const RIGHT          = DO_MOVE | convert(Action, 1<<7);  # 10000001

const UP_LEFT        = UP   | LEFT ;                     # 01010001
const UP_RIGHT       = UP   | RIGHT;                     # 10010001
const DOWN_LEFT      = DOWN | LEFT ;                     # 01100001
const DOWN_RIGHT     = DOWN | RIGHT;                     # 10100001

const moveActions = [UP, DOWN, LEFT, RIGHT,
                     UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT];

const validActions = [DO_NOTHING,
                      DO_LOAD, DO_UNLOAD, DO_RANDOM_MOVE,
                      UP, DOWN, LEFT, RIGHT,
                      UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT];

isMoveAction(a::Action) = (a & DO_MOVE) == DO_MOVE;

δy(a::Action) = (a & UP)   > DO_MOVE ? -1 : (a & DOWN)  > DO_MOVE ? 1 : 0;
δx(a::Action) = (a & LEFT) > DO_MOVE ? -1 : (a & RIGHT) > DO_MOVE ? 1 : 0;

nextCell(row::Integer, column::Integer, action::Action) =
    min(HEIGHT, max(1, row    + δy(action))),
    min(WIDTH,  max(1, column + δx(action)));

const ActionToChar = Dict{Action, Char}(DO_NOTHING     => char('\u2716'),
                                        DO_LOAD        => char('\u27f0'),
                                        DO_UNLOAD      => char('\u27f1'),
                                        DO_RANDOM_MOVE => char('?'),
                                        UP             => char('\u21d1'),
                                        DOWN           => char('\u21d3'),
                                        LEFT           => char('\u21d0'),
                                        RIGHT          => char('\u21d2'),
                                        UP_LEFT        => char('\u21d6'),
                                        UP_RIGHT       => char('\u21d7'),
                                        DOWN_LEFT      => char('\u21d9'),
                                        DOWN_RIGHT     => char('\u21d8'))

# -----------------------------------------------------------------------------

#= Signal =#

#=
    +---+---+---+---+---+---+---+---+
    | MINE SIGNAL   | DEPOSIT SIGNAL|
    +---+---+---+---+---+---+---+---+
=#

const NO_SIGNAL = convert(Signal, 0);
const NORTH     = convert(Signal, 1<<0);
const SOUTH     = convert(Signal, 1<<1);
const EAST      = convert(Signal, 1<<2);
const WEST      = convert(Signal, 1<<3);

const R4      = convert(Signal, 1 | 1<<1 | 1<<2 | 1<<3);

signalInfo(s::Signal, oldS::Signal) =
    ( s== NO_SIGNAL) ? (oldS & R4) : ((s<<4) | s);

const SignalToChar = Dict{Signal, Char}(NO_SIGNAL    => char(' '),
                                        WEST         => char('\u2190'),
                                        NORTH        => char('\u2191'),
                                        EAST         => char('\u2192'),
                                        SOUTH        => char('\u2193'),
                                        NORTH | WEST => char('\u2196'),
                                        NORTH | EAST => char('\u2197'),
                                        SOUTH | WEST => char('\u2199'),
                                        SOUTH | EAST => char('\u2198'));

# -----------------------------------------------------------------------------

#= Cartesian stuff =#

const ROW    = 1;
const COLUMN = 2;

sqEuclid(r1, c1, r2, c2) = (r1 - r2) * (r1 - r2) + (c1 - c2) * (c1 - c2);

# ---------------------------------------

type GlobalState
    board::Array{CellInfo, 2}
    warehouses::Array{Int64, 2}
    mines::Array{Int64, 2}
    agents::Array{Int64, 2}
    wSignal::Array{Signal, 2}
    mSignal::Array{Signal, 2}
    agentsMap::Array{CellInfo, 2}
end

# ---------------------------------------

function placeBuildings()
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
                warehouseIdx = warehouseIdx + 1;
                warehouses[ROW, warehouseIdx] = r;
                warehouses[COLUMN, warehouseIdx] = c;
            else
                mineIdx = mineIdx + 1;
                mines[ROW, mineIdx] = r;
                mines[COLUMN, mineIdx] = c;
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

function computeSignal(board::Array{CellInfo,2},
                       warehouses::Array{Int64,2}, mines::Array{Int64,2})
    mSignal = fill(NO_SIGNAL, size(board));
    wSignal = fill(NO_SIGNAL, size(board));
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
            bestDistance = Inf;
            mIdx = 0;
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

function initialState()
    board = fill(NOTHING, HEIGHT, WIDTH);
    # Put warehouses and mines
    warehouses, mines = placeBuildings();
    for w in 1:WAREHOUSES_NO
        board[warehouses[ROW, w], warehouses[COLUMN, w]] = WAREHOUSE;
    end
    for m in 1:MINES_NO
        board[mines[ROW, m], mines[COLUMN, m]] = MINE;
    end
    # Put agents
    agents = zeros(Int64, 2, AGENTS_NO);
    agentsMap = zeros(Int64, HEIGHT, WIDTH);
    for a in 1:AGENTS_NO
        agents[ROW, a]    = rand(1:HEIGHT)
        agents[COLUMN, a] = rand(1:WIDTH)
        while (board[agents[ROW, a], agents[COLUMN, a]] != NOTHING)
            agents[ROW, a]    = rand(1:HEIGHT)
            agents[COLUMN, a] = rand(1:WIDTH)
        end
        board[agents[ROW, a], agents[COLUMN, a]] = EMPTY;
        agentsMap[agents[ROW, a], agents[COLUMN, a]] = a;
    end
    # Compute signals
    wSignal, mSignal = computeSignal(board, warehouses, mines);
    # Pack everything
    return GlobalState(board, warehouses, mines, agents, wSignal, mSignal,
                       agentsMap);
end

function perceiveMap(gs::GlobalState,
                     a::Int64,
                     oldState::AgentState = [NO_SIGNAL, NO_SIGNAL])
    const RADIUS = trunc(Int64, AGENT_RANGE);
    const diameter = RADIUS * 2 + 1;
    const mid = RADIUS + 1;
    const row = gs.agents[ROW,a];
    const col = gs.agents[COLUMN,a];

    state = fill(NOTHING, RADIUS*2+1, RADIUS*2+1);

    for c in 1:diameter, r in 1:diameter
        if sqEuclid(r, c, mid, mid) <= AGENT_RANGE * AGENT_RANGE
            state[r, c] =
                ((1 <= col-mid+c <= WIDTH) && (1 <= row-mid+r <= HEIGHT)) ?
            gs.board[row-mid+r,col-mid+c] : WALL
        end
    end

    const wSignal = signalInfo(gs.wSignal[gs.agents[ROW,a],gs.agents[COLUMN,a]],
                               oldState[1]);
    const mSignal = signalInfo(gs.mSignal[gs.agents[ROW,a],gs.agents[COLUMN,a]],
                               oldState[2]);
    return vcat(wSignal, mSignal, state[:]);
end

function sortAgents(gs::GlobalState, actions::Array{Action, 1})
    sortedAgents = zeros(Int64, AGENTS_NO);
    crtIdx = 1;

    isBlocked = fill(true, AGENTS_NO);
    waitsOn = zeros(Int64, AGENTS_NO);

    # First put in the finall list all agents that do not wait on others to move
    for i in 1:AGENTS_NO
        if isMoveAction(actions[i])
            const crtRow = gs.agents[ROW,i];
            const crtColumn = gs.agents[COLUMN,i];
            nextRow, nextColumn = nextCell(crtRow, crtColumn, actions[i]);
            if (((nextRow != crtRow) || (nextColumn != crtColumn)) &&
                isAgent(gs.board[nextRow, nextColumn]) &&
                isBlocked[gs.agentsMap[nextRow, nextColumn]])
                waitsOn[i] = gs.agentsMap[nextRow, nextColumn];
                continue;
            end
        end
        sortedAgents[crtIdx] = i;
        crtIdx = crtIdx + 1;
        isBlocked[i] = false;
    end

    # Put agents that wait for agents already put in the final list
    foundOne = true;
    while foundOne && crtIdx <= AGENTS_NO
        foundOne = false;
        for i in find(isBlocked)
            if isBlocked[waitsOn[i]] == false
                foundOne = true
                sortedAgents[crtIdx] = i;
                crtIdx = crtIdx + 1;
                isBlocked[i] = false;
            end
        end
    end
    return vcat(sortedAgents[1:(crtIdx-1)], find(isBlocked));
end

function doLoad(gs::GlobalState, ag::Int64)
    row = gs.agents[ROW, ag];
    col = gs.agents[COLUMN, ag];
    if isEmpty(gs.board[row, col])
        for m in 1:MINES_NO
            if sqEuclid(gs.mines[ROW,m], gs.mines[COLUMN,m], row, col) <= 2
                gs.board[row,col] = LOADED;
                return
            end
        end
    end
    nothing
end

function doUnload(gs::GlobalState, ag::Int64)
    row = gs.agents[ROW, ag];
    col = gs.agents[COLUMN, ag];
    if isLoaded(gs.board[row, col])
        for w in 1:WAREHOUSES_NO
            if sqEuclid(gs.warehouses[ROW,w], gs.warehouses[COLUMN,w],
                        row, col) <= 2
                gs.board[row, col] = EMPTY;
                return w;
            end
        end
    end
    return 0;
end

function doRandomMove(gs::GlobalState, ag::Int64)
    doMove(gs, ag, moveActions[rand(1:end)])
    nothing
end

function doMove(gs::GlobalState, ag::Int64, action::Action)
    const row = gs.agents[ROW, ag];
    const column = gs.agents[COLUMN, ag];
    const newRow, newColumn = nextCell(row, column, action);
    if gs.board[newRow, newColumn] == NOTHING
        gs.board[newRow, newColumn] = gs.board[row, column];
        gs.board[row, column] = NOTHING;
        gs.agents[ROW, ag] = newRow;
        gs.agents[COLUMN, ag] = newColumn;
        gs.agentsMap[row, column] = NOTHING;
        gs.agentsMap[newRow, newColumn] = ag;
    end
    nothing
end

reward(dropCount::Int64) = 2.5 ^ dropCount - 1;

function doActions(gs::GlobalState, actions::Array{Action, 1})
    rewards        = zeros(Float64, AGENTS_NO);
    rewarders      = zeros(Int64, AGENTS_NO);
    warehouseDrops = zeros(Int64,   WAREHOUSES_NO);

    for ag in sortAgents(gs, actions)
        if actions[ag] != DO_NOTHING
            if actions[ag] == DO_LOAD
                doLoad(gs, ag);
            elseif actions[ag] == DO_UNLOAD
                wIdx = doUnload(gs, ag);
                if wIdx > 0
                    warehouseDrops[wIdx] = warehouseDrops[wIdx] + 1;
                    rewarders[ag] = wIdx;
                end
            elseif actions[ag] == DO_RANDOM_MOVE
                doRandomMove(gs, ag);
            else
                doMove(gs, ag, actions[ag]);
            end
        end
    end

    for ag in 1:AGENTS_NO
        if rewarders[ag] > 0
            if warehouseDrops[rewarders[ag]] > 0
                rewards[ag] = reward(warehouseDrops[rewarders[ag]]);
            end
        end
    end

    return rewards, warehouseDrops;
end

function printState(state::AgentState)
    wSignal = state[1];
    crtWSignal = wSignal>>4;
    oldWSignal = (wSignal<<4)>>4;
    mSignal = state[2];
    crtMSignal = mSignal>>4;
    oldWSignal = (mSignal<<4)>>4;
    println("W: $(SignalToChar[crtWSignal]) - $(SignalToChar[crtWSignal])");
    println("M: $(SignalToChar[crtMSignal]) - $(SignalToChar[crtMSignal])");
    l = int(sqrt(length(state[3:end])));
    printBoard(reshape(state[3:end], l, l));
end

function printSignalMap(signalMap::Array{Signal, 2})
    for col in 1:(size(signalMap,2)+2)
        print(char('\u2588'));
    end
    println();
    for row in 1:size(signalMap,1)
        print(char('\u2588'));
        for col in 1:size(signalMap,2)
            print(SignalToChar[signalMap[row, col]]);
        end
        println(char('\u2588'));
    end
    for col in 1:(size(signalMap,2)+2)
        print(char('\u2588'));
    end
    println();
end

function printBoard(board::Array{CellInfo, 2})
    for col in 1:(size(board,2)+2)
        print(char('\u2588'));
    end
    println();
    for row in 1:size(board,1)
        print(char('\u2588'));
        for col in 1:size(board,2)
            print(CellInfoToChar[board[row, col]]);
        end
        println(char('\u2588'));
    end
    for col in 1:(size(board,2)+2)
        print(char('\u2588'));
    end
    println();
end

end
