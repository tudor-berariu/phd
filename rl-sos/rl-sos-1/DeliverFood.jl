#!/usr/local/bin/julia

#=
    Both actions and cells on the map are going to be represented on 8 bits.
    The state of a cell will use the first bit to discriminate between agents
    and buildings.
=#

typealias Action Int8;
typealias CellInfo Int8;
typealias AgentState Array{CellInfo, 2};

const UP          = convert(Action, 1<<0);
const DOWN        = convert(Action, 1<<1);
const LEFT        = convert(Action, 1<<2);
const RIGHT       = convert(Action, 1<<3);
const LOAD        = convert(Action, 1<<4);
const UNLOAD      = convert(Action, 1<<5);

const NOTHING = UP | DOWN;
const UP_LEFT = UP | LEFT;
const UP_RIGHT = UP | RIGHT;
const DOWN_LEFT = DOWN | LEFT;
const DOWN_RIGHT = DOWN | RIGHT;

const IS_LOADED   = convert(CellInfo, 1<<6);
const IS_BUILDING = convert(CellInfo, 1<<7);

const valid_actions = Action[NOTHING,
                             UP, DOWN, LEFT, RIGHT,
                             UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT,
                             LOAD, UNLOAD];

dy(action::Action) = action & UP > 0 ? -1 : action & DOWN > 0 ? 1 : 0;
dx(action::Action) = action & LEFT > 0 ? -1 : action & RIGHT > 0 ? 1 : 0;

const RADIUS = 2;

const ROW = 1
const COLUMN = 2

# ------------------------------------------------------------------------------

type GlobalState
    height::Integer
    width::Integer
    board::Array{CellInfo, 2}
    deposits::Array{Int64, 2}
    mines::Array{Int64, 2}
    N::Int64
    agents::Array{Int64, 2}
    K::Int64
    last_K::Array{Float64, 2}
    j::Int64
end

const MINE = convert(CellInfo, IS_BUILDING | LOAD);
const DEPOSIT = convert(CellInfo, IS_BUILDING | UNLOAD);

function init_scenario(height::Int64,
                       width::Int64,
                       deposits::Array{Int64,2},
                       mines::Array{Int64,2},
                       N::Int64,
                       K::Int64)
    board = zeros(width, height)
    # Mark deposits on board
    for d in 1:size(deposits,2)
        board[deposits[ROW,d], deposits[COLUMN,d]] = DEPOSIT
    end
    last_K = zeros(Float64, K, size(deposits, 2));
    # Mark mines on board
    for m in 1:size(mines,2)
        board[mines[ROW,m], mines[COLUMN,m]] = MINE
    end
    # Spread agents on board
    agents = zeros(Int8, 2, N);
    agents[ROW,:] = rand(1:height, N);
    agents[COLUMN,:] = rand(1:width, N);
    for i in 1:N
        while board[agents[ROW,i], agents[COLUMN,i]] != 0
            agents[ROW,i] = rand(1:height)
            agents[COLUMN,i] = rand(1:width)
        end
        board[agents[ROW,i], agents[COLUMN,i]] = UP
    end
    # Put everything together
    return GlobalState(height, width, board,
                       deposits, mines, N, agents, K, last_K, 1);
end

function perceive(S::GlobalState, i::Integer)
    t_idx = S.agents[ROW, i] - RADIUS;
    b_idx = S.agents[ROW, i] + RADIUS;
    l_idx = S.agents[COLUMN, i] - RADIUS;
    r_idx = S.agents[COLUMN, i] + RADIUS;

    t_rows = max(0,  1 - t_idx);
    b_rows = max(0, b_idx - S.height);
    l_cols = max(0, 1 - l_idx);
    r_cols = max(0, r_idx - S.width);

    return vcat(zeros(CellInfo, t_rows, 2 * RADIUS + 1),
                hcat(zeros(CellInfo, 2 * RADIUS + 1 - t_rows - b_rows, l_cols),
                     S.board[max(1, t_idx):min(S.height, b_idx),
                             max(1, l_idx):min(S.width, r_idx)],
                     zeros(CellInfo, 2 * RADIUS + 1 - t_rows - b_rows, r_cols)),
                zeros(CellInfo, b_rows, 2 * RADIUS + 1));
end

function do_load(S::GlobalState, ag::Integer)
    row = S.agents[ROW, ag];
    col = S.agents[COLUMN, ag];
    S.board[row, col] = (S.board[row, col] & IS_LOADED) | LOAD;
    if S.board[row, col] & IS_LOADED == 0
        for m in 1:size(S.mines, 2)
            if abs(row - S.mines[ROW, m]) + abs(col - S.mines[COLUMN, m]) == 1
                # The LOAD action succeedes
                S.board[row, col] = IS_LOADED | LOAD
                break
            end
        end
    end
    nothing
end

function do_unload(S::GlobalState, ag::Integer)
    reward = 0.0;
    row = S.agents[ROW, ag];
    col = S.agents[COLUMN, ag];
    S.board[row, col] = (S.board[row, col] & IS_LOADED) | UNLOAD;
    if S.board[row, col] & IS_LOADED > 0
        for d in 1:size(S.deposits, 2)
            if abs(row-S.deposits[ROW,d])+ abs(col-S.deposits[COLUMN,d]) == 1
                # The UNLOAD action succeeds
                S.board[row, col] = UNLOAD;
                S.last_K[S.j, d] = S.last_K[S.j, d] + 1;
                reward = sum(S.last_K[:, d]);
                break;
            end
        end
    end
    return reward;
end

function do_move(S::GlobalState, ag::Integer, action::Action)
    row = S.agents[ROW, ag];
    col = S.agents[COLUMN, ag];
    if action & NOTHING == NOTHING
        S.board[row, col] = (S.board[row, col] & IS_LOADED) | action;
    end
    δrows = dy(action)
    δcols = dx(action)
    new_row = min(S.height, max(1, row + δrows))
    new_col = min(S.width, max(1, col + δcols))
    if S.board[new_row, new_col] != 0
        S.board[row, col] = (S.board[row, col] & IS_LOADED) | action;
    else
        S.board[new_row, new_col] = (S.board[row, col] & IS_LOADED) | action;
        S.board[row, col] = zero(S.board[row, col]);
        S.agents[ROW, ag] = new_row;
        S.agents[COLUMN, ag] = new_col;
    end
    nothing
end


function get_best(Qs::Dict{AgentState, Dict{Action, Float64}},
                  state::AgentState)
    As_Qs = get!(Qs, state, Dict{Action, Float64}());
    return reduce((best, k) -> As_Qs[k] > best[1] ? tuple(As_Qs[k], k) : best,
                  tuple(0.0, valid_actions[rand(1:end)]),
                  keys(As_Qs));
end

function choose_action(s::AgentState,
                       Qs::Dict{AgentState, Dict{Action, Float64}},
                       ϵ::Float64 = 0.01)
    return rand() < ϵ ? valid_actions[rand(1:end)] : get_best(Qs, s)[2];
end

function update_qs(Qs::Dict{AgentState, Dict{Action, Float64}},
                   s1::AgentState, a::Action, reward::Float64, s2::AgentState,
                   γ::Float64, α::Float64)
    As_Qs = get!(Qs, s1, Dict{Action, Float64}());
    oldQ = get!(As_Qs, a, 0.0);
    As_Qs[a] = (1-α) * oldQ + α * (reward + γ * get_best(Qs, s2)[1]);
    nothing
end

function learn(S::GlobalState, episodes_no::Int64)
    Qs = Array(Dict{AgentState, Dict{Action, Float64}}, S.N);
    for i in 1:S.N
        Qs[i] = Dict{AgentState, Dict{Action, Float64}}();
    end
    #-------------------
    actions = Array(Action, S.N);
    prev_states = Array(AgentState, S.N);
    states = Array(AgentState, S.N);
    rewards = Array(Float64, S.N);
    for season in 1:100000
        S = init_scenario(S.height, S.width, S.deposits, S.mines, S.N, S.K);
        for i in 1:S.N
            states[i] = perceive(S, i);
        end
        # -------------------
        for ep in 1:episodes_no
            #= All agents select their actions. =#
            for ag in 1:S.N
                actions[ag] = choose_action(states[ag], Qs[ag]);
                prev_states[ag] = deepcopy(states[ag]);
            end
            #=
                All deposits put a zero on current time
            =#
            for d in 1:size(S.deposits, 2)
                S.last_K[S.j, d] = 0.0;
            end
            #=
                All agents perform their actions and the board changes.
            =#
            for i in shuffle([1:S.N])
                rewards[i] = 0.0;
                if actions[i] & LOAD > 0
                    do_load(S, i);
                elseif actions[i] & UNLOAD > 0
                    rewards[i] = do_unload(S, i);
                else
                    do_move(S, i, actions[i]);
                end
            end
            #=
                Increment j
            =#
            S.j = S.j % S.K + 1;
            #=
            Save states to prev_states
            =#
            for i = 1:S.N
                states[i] = perceive(S, i);
                update_qs(Qs[i], prev_states[i],
                          actions[i], rewards[i], states[i],
                          0.95, 0.05);
            end
            if ep % 1000 == 0
                scores = Array(Float64, 50);
                for es in 1:20
                    S1 = init_scenario(S.height, S.width,
                                       S.deposits, S.mines, S.N, S.K);
                    scores[es] = evaluate(S1, 300, Qs);
                end
                println("season=", season, ",",
                        "\tepisode=", ep, ",",
                        "\tstates=", mean(map(i->length(keys(Qs[i])), 1:S.N)),
                        "\tscore=", mean(scores));
            end
        end
    end
    #println(Qs)
    return Qs;
end


function evaluate(S::GlobalState,
                  episodes_no::Int64,
                  Qs::Array{Dict{AgentState, Dict{Action, Float64}},1})
    #-------------------
    actions = Array(Action, S.N);
    states = Array(AgentState, S.N);
    score = 0;
    for i in 1:S.N
        states[i] = perceive(S, i);
    end
    # -------------------
    for ep in 1:episodes_no
        #= All agents select their actions. =#
        for ag in 1:S.N
            actions[ag] = get_best(Qs[ag], states[ag])[2];
        end
        #=
            All deposits put a zero on current time
        =#
        for d in 1:size(S.deposits, 2)
            S.last_K[S.j, d] = 0.0;
        end
        #=
            All agents perform their actions and the board changes.
        =#
        for i in shuffle([1:S.N])
            if actions[i] & LOAD > 0
                do_load(S, i);
            elseif actions[i] & UNLOAD > 0
                do_unload(S, i);
            else
                do_move(S, i, actions[i]);
            end
        end
        #=
            Save states to prev_states
        =#
        for i = 1:S.N
            states[i] = perceive(S, i);
        end
        score = score + sum(S.last_K[S.j, :]);
    end
    return score;
end


const to_screen = Dict{Int8, Char}([NOTHING, UP, UP_RIGHT, RIGHT,
                                    DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT,
                                    UP_LEFT,
                                    NOTHING | IS_LOADED, UP | IS_LOADED,
                                    UP_RIGHT | IS_LOADED, RIGHT | IS_LOADED,
                                    DOWN_RIGHT | IS_LOADED, DOWN | IS_LOADED,
                                    DOWN_LEFT | IS_LOADED, LEFT | IS_LOADED,
                                    UP_LEFT | IS_LOADED, DEPOSIT, MINE, 0,
                                    LOAD | IS_LOADED, UNLOAD | IS_LOADED,
                                    LOAD, UNLOAD],
                                    [char('\u25a1'), char('\u2191'),
                                     char('\u2197'), char('\u2192'),
                                     char('\u2198'), char('\u2193'),
                                     char('\u2199'), char('\u2190'),
                                     char('\u2196'), char('\u25a0'),
                                     char('\u21d1'), char('\u21d7'),
                                     char('\u21d2'), char('\u21d8'),
                                     char('\u21d3'), char('\u21d9'),
                                     char('\u21d0'), char('\u21d6'),
                                     char('\u25c7'), char('\u25c6'), ' ',
                                     char('\u25a0'), char('\u25a0'),
                                     char('\u25a1'), char('\u25a1')]);

function display_board(board)
    for row in 1:size(board,1)
        for col in 1:size(board,2)
            print(to_screen[board[row, col]]);
        end
        println();
    end
end

S = init_scenario(6, 6, reshape([2, 2],2,1), reshape([5, 5],2,1), 3, 8);
println(typeof(S.N))
learn(S, 100000);
