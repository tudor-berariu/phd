#!/usr/local/bin/julia

using HDF5, JLD
using Cairo;
using DataFrames;
#using Gadfly;

#=

    In this scenario a couple of agents need to learn how to deliver
    gold from a mine to a deposit. Both mines and deposits emit radio
    signals so agents know in which direction these objectives are.
    Agents get an exponentialy higher reward if they drop the gold at
    the deposit at the same time.
    Will they learn to synchronize?

    Reward = min(2^No. of. dropping agents, 256)

=#

# -----------------------------------------------------------------------------

#= Global Paramteres that control the experiment. =#

const HEIGHT           = 10;
const WIDTH            = 10;
const AGENTS_NO        = 1;
const WAREHOUSES_NO    = 1;
const MINES_NO         = 1;
const RANGE            = 15;
const AGENT_RANGE      = 1;
const MIN_DIST         = 3;

const SEASONS_NO       = 10000;
const EPISODES_NO      = 5000;

const EVAL_EVERY       = 200;
const EVAL_SEASONS_NO  = 10;
const EVAL_EPISODES_NO = 1000;

# -----------------------------------------------------------------------------

typealias CellInfo Uint8;
typealias RadioSignal Uint8;
typealias AgentState Array{CellInfo, 1};
typealias Action Uint8;

# -----------------------------------------------------------------------------

#= CellInfo =#

#=
    +---+---+---+---+---+---+---+---+
    |   |   |   |   |   |   |   |N/S|
    +---+---+---+---+---+---+---+---+
=#
const NOTHING   = convert(CellInfo, 0);
const SOMETHING = convert(CellInfo, 1);

isNothing(cell::CellInfo) = cell == 0;
isSomething(cell::CellInfo) = cell > 0;

#=
    +---+---+---+---+---+---+---+---+
    |   |   |   |   | W | B | A | 1 |
    +---+---+---+---+---+---+---+---+
=#
const AGENT    = SOMETHING | convert(CellInfo, 1<<1);
const BUILDING = SOMETHING | convert(CellInfo, 1<<2);
const WALL     = SOMETHING | convert(CellInfo, 1<<3);

isAgent(cell::CellInfo) = (cell | AGENT) == AGENT;
isBuilding(cell::CellInfo) = (cell | BUILDING) == BUILDING;
isWall(cell::CellInfo) = (cell | WALL) == WALL;

#=
    Agent info
    +---+---+---+---+---+---+---+---+
    |   |   | E | L | 0 | 0 | 1 | 1 |
    +---+---+---+---+---+---+---+---+
=#

const LOADED = SOMETHING | AGENT | convert(CellInfo, 1<<4)
const EMPTY  = SOMETHING | AGENT | convert(CellInfo, 1<<5)

isLoaded(agent::CellInfo) = (agent | LOADED) == LOADED;
isEmpty(agent::CellInfo) = (agent | EMPTY) == EMPTY;

#=
    Building info
    +---+---+---+---+---+---+---+---+
    |   |   | M | W | 0 | 1 | 0 | 1 |
    +---+---+---+---+---+---+---+---+
=#

const WAREHOUSE = SOMETHING | BUILDING | convert(CellInfo, 1<<4);
const MINE      = SOMETHING | BUILDING | convert(CellInfo, 1<<5);

isWarehouse(building::CellInfo) = (building | WAREHOUSE) == WAREHOUSE;
isMine(building::CellInfo) = (building | MINE) == MINE;

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

isMoveAction(action::Action) = (action | DO_MOVE) == DO_MOVE;

δy(action::Action) =
    (action & UP) > DO_MOVE ? -1 : (action & DOWN) > DO_MOVE ? 1 : 0;
δx(action::Action) =
    (action & LEFT) > 1 ? -1 : (action & RIGHT) > 1 ? 1 : 0;

function nextCell(row::Integer, column::Integer, action::Action)
    δRows = δy(action);
    δColumns = δx(action);
    newRow = min(HEIGHT, max(1, row + δRows));
    newColumn = min(WIDTH, max(1, column + δColumns));
    return newRow, newColumn;
end

# -----------------------------------------------------------------------------

#= RadioInfo =#

#=
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

# -----------------------------------------------------------------------------

const ROW    = 1;
const COLUMN = 2;

sqEuclid(r1, c1, r2, c2) = (r1 - r2) * (r1 - r2) + (c1 - c2) * (c1 - c2);

# ---------------------------------------

type GlobalState
    board::Array{CellInfo, 2}
    warehouses::Array{Int64, 2}
    mines::Array{Int64, 2}
    agents::Array{Int64, 2}
    wSignal::Array{RadioSignal, 2}
    mSignal::Array{RadioSignal, 2}
    agentsMap::Array{CellInfo, 2}
end

type Evaluation
    scores::Array{Int64, 2}
    statesNo::Array{Int64, 2}
    drops::Array{Int64, 2}
end

# ---------------------------------------

function placeBuildings()
    warehouses = zeros(Integer, 2, WAREHOUSES_NO);
    mines = zeros(Integer, 2, MINES_NO);

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
    agents = zeros(Integer, 2, AGENTS_NO);
    agentsMap = zeros(Integer, HEIGHT, WIDTH);
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
    wSignal, mSignal = computeRadioSignal(board, warehouses, mines);
    # Pack everything
    return GlobalState(board, warehouses, mines, agents, wSignal, mSignal,
                       agentsMap);
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
    state = [sqEuclid(r, c, mid, mid) <= AGENT_RANGE * AGENT_RANGE ?
             NOTHING : state[r, c] for c in 1:diameter, r in 1:diameter];

    wSignal = newSignal(gs.wSignal[gs.agents[ROW, a], gs.agents[COLUMN, a]],
                        oldState[1]);
    mSignal = newSignal(gs.mSignal[gs.agents[ROW, a], gs.agents[COLUMN, a]],
                        oldState[2]);
    return vcat(wSignal, mSignal, state[:]);
end

function sortAgents(gs::GlobalState, actions::Array{Action, 1})
    sortedAgents = zeros(Integer, AGENTS_NO);
    crtIdx = 1;
    leftAgents = Array(Integer, 0);
    waitsOn = zeros(Integer, AGENTS_NO);

    # First put in the finall list all agents that do not wait on others to move
    for i in 1:AGENTS_NO
        if isMoveAction(actions[i])
            crtRow = gs.agents[ROW,i];
            crtColumn = gs.agents[COLUMN,i];
            nextRow, nextColumn = nextCell(crtRow, crtColumn, actions[i]);
            if (((nextRow != crtRow) || (nextColumn != crtColumn)) &&
                isAgent(gs.board[nextRow, nextColumn]))
                push!(leftAgents, i);
                waitsOn[i] = gs.agentsMap[nextRow, nextColumn];
                continue;
            end
        end
        sortedAgents[crtIdx] = i;
        crtIdx = crtIdx + 1;
    end

    # Put agents that wait for agents already put in the final list
    foundOne = true;
    while foundOne && crtIdx <= AGENTS_NO
        tmp = Array(Integer, 0)
        foundOne = false;
        for i in 1:length(leftAgents)
            canPerform = false;
            for j in length(sortedAgents):-1:1
                if sortedAgents[j] == waitsOn[i]
                    foundOne = true;
                    canPerform = true;
                    break;
                end
            end
            if canPerform
                sortedAgents[crtIdx] = i;
                crtIdx = crtIdx + 1;
            else
                push!(tmp, i);
            end
        end
        leftAgents = deepcopy(tmp);
    end
    return vcat(sortedAgents, leftAgents);
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
    if isLoaded(gs.board[row, col])
        if (((row > 1) && isWarehouse(gs.board[row-1,col])) ||
            ((row < HEIGHT) && isWarehouse(gs.board[row+1,col])) ||
            ((col < WIDTH) && isWarehouse(gs.board[row,col+1])) ||
            ((col > 1) && isWarehouse(gs.board[row,col-1])))
            gs.board[row, col] = EMPTY;
            for w in 1:WAREHOUSES_NO
                if sqEuclid(gs.warehouses[ROW,w], gs.warehouses[COLUMN,w],
                            row, col) <= 1
                    return w;
                end
            end
        end
    end
    return 0;
end

function doRandomMove(gs::GlobalState, ag::Integer)
    doMove(gs, ag, moveActions[rand(1:end)])
    nothing
end

function doMove(gs::GlobalState, ag::Integer, action::Action)
    row = gs.agents[ROW, ag];
    column = gs.agents[COLUMN, ag];
    newRow, newColumn = nextCell(row, column, action);
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

reward(dropCount::Integer) = 2.0 ^ dropCount - 1;

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
    #= Q values =#
    Qs = Array(Dict{AgentState, Dict{Action, Float64}}, AGENTS_NO);
    for ag in 1:AGENTS_NO
        Qs[ag] = Dict{AgentState, Dict{Action, Float64}}();
    end
    #= Actions, States, Rewards =#
    actions = Array(Action, AGENTS_NO);
    prevStates = Array(AgentState, AGENTS_NO);
    for ag in 1:AGENTS_NO
        prevStates[ag] = vcat(NO_SIGNAL, NO_SIGNAL);
    end
    states = Array(AgentState, AGENTS_NO);
    #= Evaluation =#
    eval = Evaluation(zeros(Int64, EVAL_SEASONS_NO, div(SEASONS_NO,EVAL_EVERY)),
                      zeros(Int64, AGENTS_NO, div(SEASONS_NO,EVAL_EVERY)),
                      zeros(Int64, 8, div(SEASONS_NO,EVAL_EVERY)));
    # ---
    for season in 1:SEASONS_NO
        #println("S$(season)");
        gs = initialState()
        for ag in 1:AGENTS_NO
            states[ag] = perceiveMap(gs, prevStates[ag], ag);
        end
        max_ϵ = log_decay(0.01, 0.3, season, SEASONS_NO);
        for episode in 1:EPISODES_NO
            # Choose actions
            ϵ = log_decay(0.003, max_ϵ, episode, EPISODES_NO);
            for ag in 1:AGENTS_NO
                actions[ag] = chooseAction(states[ag], Qs[ag], ϵ);
                prevStates[ag] = deepcopy(states[ag]);
            end
            # Do actions
            rewards = zeros(Float64, AGENTS_NO);
            rewarders = zeros(Integer, AGENTS_NO);
            wDrops = zeros(Int64, WAREHOUSES_NO);
            for ag in sortAgents(gs, actions)
                if actions[ag] == DO_NOTHING
                    continue
                end
                if actions[ag] == DO_LOAD
                    doLoad(gs, ag);
                elseif actions[ag] == DO_UNLOAD
                    wIdx = doUnload(gs, ag);
                    if wIdx > 0
                        wDrops[wIdx] = wDrops[wIdx] + 1;
                    end
                    rewarders[ag] = wIdx;
                elseif actions[ag] == DO_RANDOM_MOVE
                    doRandomMove(gs, ag);
                else
                    doMove(gs, ag, actions[ag]);
                end
            end
            for ag in 1:AGENTS_NO
                if rewarders[ag] > 0
                    if wDrops[rewarders[ag]] > 0
                        rewards[ag] = reward(wDrops[rewarders[ag]]);
                    end
                end
                states[ag] = perceiveMap(gs, prevStates[ag], ag);
                update_qs(Qs[ag], prevStates[ag],
                          actions[ag], rewards[ag], states[ag],
                          0.97, 0.05);
            end
        end
        if mod(season, EVAL_EVERY) == 0
            evaluate(Qs, eval, season);
        end
    end
end

function evaluate(Qs::Array{Dict{AgentState, Dict{Action, Float64}}, 1},
                  eval::Evaluation,
                  season::Integer)
    const idx = div(season, EVAL_EVERY);
    println(idx);
    #= Number of states =#
    for ag in 1:AGENTS_NO
        eval.statesNo[ag, idx] = length(keys(Qs[ag]));
    end

    #= Actions, States, Rewards =#
    actions = Array(Action, AGENTS_NO);
    prevStates = Array(AgentState, AGENTS_NO);
    for ag in 1:AGENTS_NO
        prevStates[ag] = vcat(NO_SIGNAL, NO_SIGNAL);
    end
    states = Array(AgentState, AGENTS_NO);

    #= Run seasons =#
    for evalSeason in 1:EVAL_SEASONS_NO
        gs = initialState();
        #displayBoard(gs.board);
        for ag in 1:AGENTS_NO
            states[ag] = perceiveMap(gs, prevStates[ag], ag);
        end
        for episode in 1:EPISODES_NO
            #= Choose actions =#
            for ag in 1:AGENTS_NO
                actions[ag] = getBest(Qs[ag], states[ag])[2];
            end
            prevStates = deepcopy(states);
            #= Do actions =#
            warehouseDrops = zeros(Int64, WAREHOUSES_NO);
            for ag in sortAgents(gs, actions)
                if actions[ag] == DO_NOTHING
                    continue
                end
                if actions[ag] == DO_LOAD
                    println("!");
                    doLoad(gs, ag);
                elseif actions[ag] == DO_UNLOAD
                    wIdx = doUnload(gs, ag);
                    println("!!!");
                    if wIdx > 0
                        print("<<<<");
                        warehouseDrops[wIdx] = warehouseDrops[wIdx] + 1;
                    end
                    println();
                elseif actions[ag] == DO_RANDOM_MOVE
                    doRandomMove(gs, ag);
                else
                    doMove(gs, ag, actions[ag]);
                end
            end
            #= See scores =#
            for w in 1:WAREHOUSES_NO
                if warehouseDrops[w] > 0
                    eval.scores[evalSeason,idx] =
                        eval.scores[evalSeason,idx] + reward(warehouseDrops[w]);
                    eval.drops[wDrops[w],idx] = eval.drops[wDrops[w],idx] + 1;
                end
            end
        end
    end
    #= Save plots =#
    # Plot number of states
    dfStates = DataFrame(Season = EVAL_EVERY:EVAL_EVERY:season,
                         Max = 1.0 .* maximum(eval.statesNo[:,1:idx],1)[:],
                         Min = 1.0.*minimum(eval.statesNo[:,1:idx],1)[:],
                         Mean = mean(eval.statesNo[:,1:idx],1)[:]);
    draw(PDF("results/states.pdf", 6inch, 3inch),
         plot(dfStates, x=:Season, y=:Mean, ymax=:Max, ymin=:Min,
              Geom.line, Geom.ribbon,
              Guide.ylabel("No. of states"),
              Guide.title("Evolution of dictionary size"),
              Guide.xlabel("Learning Season"))
         );
    # Plot score
    dfScore = DataFrame(Season = EVAL_EVERY:EVAL_EVERY:season,
                        Max = 1.0 .* maximum(eval.scores[:,1:idx], 1)[:],
                        Min = 1.0 .* minimum(eval.scores[:,1:idx], 1)[:],
                        Mean = mean(eval.scores[:,1:idx], 1)[:]);
    draw(PDF("results/scores.pdf", 6inch, 3inch),
         plot(dfScore, x=:Season, y=:Mean, ymax=:Max, ymin=:Min,
              Geom.line, Geom.ribbon,
              Guide.ylabel("Score"),
              Guide.title("Evolution of score"),
              Guide.xlabel("Learning Season"))
         );
    # Plot drops
    dfDrops=DataFrame(Season = [div(i, 8) + 1 for i=0:(idx * 8 - 1)],
                      Drops = [mod(i, 8) + 1 for i=0:(idx * 8 - 1)],
                      Count = eval.drops[1:(idx * 8)])
    #println(dfDrops);
    draw(PDF("results/drops.pdf", 6inch, 3inch),
         plot(dfDrops, x=:Season, y=:Count, color=:Drops,
              Geom.bar(position=:stack),
              Guide.yticks(ticks=[0:5:maximum(sum(eval.drops,1))]))
         );
    JLD.save("results/qs_S$season.jld", "Qs", Qs);
end

function testQs(episodesNo, label::Integer)
    Qs = JLD.load("results/qs_S$(label).jld", "Qs");
    #= Actions, States, Rewards =#
    actions = Array(Action, AGENTS_NO);
    prevStates = Array(AgentState, AGENTS_NO);
    for ag in 1:AGENTS_NO
        prevStates[ag] = vcat(NO_SIGNAL, NO_SIGNAL);
    end
    states = Array(AgentState, AGENTS_NO);
    gs = initialState();
    displayBoard(gs.board);
    sleep(2);
    for ag in 1:AGENTS_NO
        states[ag] = perceiveMap(gs, prevStates[ag], ag);
    end
    for episode in 1:episodesNo
        #= Choose actions =#
        for ag in 1:AGENTS_NO
            displayState(states[ag]);
            actions[ag] = getBest(Qs[ag], states[ag])[2];
        end
        prevStates = deepcopy(states);
        #= Do actions =#
        warehouseDrops = zeros(Int64, WAREHOUSES_NO);
        for ag in sortAgents(gs, actions)
            if actions[ag] == DO_NOTHING
                continue
            end
            if actions[ag] == DO_LOAD
                doLoad(gs, ag);
            elseif actions[ag] == DO_UNLOAD
                wIdx = doUnload(gs, ag);
                if wIdx > 0
                    warehouseDrops[wIdx] = warehouseDrops[wIdx] + 1;
                end
            elseif actions[ag] == DO_RANDOM_MOVE
                doRandomMove(gs, ag);
            else
                doMove(gs, ag, actions[ag]);
            end
        end
    end
end

const boardChar = Dict{CellInfo, Char}(EMPTY => char('\u26c4'),
                                       LOADED => char('\u26c7'),
                                       WAREHOUSE => char('\u26ea'),
                                       MINE => char('\u233e'),
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

const actionChar = Dict{Action, Char}(DO_NOTHING => char('X'),
                                      DO_LOAD => char('L'),
                                      DO_UNLOAD => char('U'),
                                      DO_RANDOM_MOVE => char('R'),
                                      UP => char('\u21d1'),
                                      DOWN => char('\u21d3'),
                                      LEFT => char('\u21d0'),
                                      RIGHT => char('\u21d2'),
                                      UP_LEFT => char('\u21d6'),
                                      UP_RIGHT => char('\u21d7'),
                                      DOWN_LEFT => char('\u21d9'),
                                      DOWN_RIGHT => char('\u21d8'))

function displayState(state::AgentState)
    wSignal = state[1];
    crtWSignal = wSignal>>4;
    oldWSignal = (wSignal<<4)>>4;
    mSignal = state[2];
    crtMSignal = mSignal>>4;
    oldWSignal = (mSignal<<4)>>4;
    println("Current state : ");
    println("$(signalChar[crtWSignal]) - $(signalChar[crtWSignal])");
    println("$(signalChar[crtMSignal]) - $(signalChar[crtMSignal])");
    l = int(sqrt(length(state[2:end])));
    printBoard(reshape(state[2:end], l, l));
    println();
end

function displaySignalMap(signalMap::Array{RadioSignal, 2})
    for col in 1:(size(signalMap,2)+2)
        print(char('\u2588'));
    end
    println();
    for row in 1:size(signalMap,1)
        print(char('\u2588'));
        for col in 1:size(signalMap,2)
            print(signalChar[signalMap[row, col]]);
        end
        println(char('\u2588'));
    end
    for col in 1:(size(signalMap,2)+2)
        print(char('\u2588'));
    end
    println();
end

function displayBoard(board::Array{CellInfo, 2})
    for col in 1:(size(board,2)+2)
        print(char('\u2588'));
    end
    println();
    for row in 1:size(board,1)
        print(char('\u2588'));
        for col in 1:size(board,2)
            print(boardChar[board[row, col]]);
        end
        println(char('\u2588'));
    end
    for col in 1:(size(board,2)+2)
        print(char('\u2588'));
    end
    println();
end

# learn()
