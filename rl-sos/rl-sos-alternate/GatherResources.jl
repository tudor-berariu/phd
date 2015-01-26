#!/usr/local/bin/julia


#=
    Here agents learn alternatively
=#

using Gadfly
using HDF5, JLD
using DataFrames
#=
    Both actions and cells on the map are going to be represented on 8 bits.
    The state of a cell will use the first bit to discriminate between agents
    and buildings.

    +---+---+---+---+---+---+---+---+
    | B | C | U | L | > | < | v | ^ |
    +---+---+---+---+---+---+---+---+
=#

typealias Action Uint8;
typealias CellInfo Uint8;
typealias AgentState Array{CellInfo, 2};

const UP          = convert(Action, 1<<0);     # 00000001
const DOWN        = convert(Action, 1<<1);     # 00000010
const LEFT        = convert(Action, 1<<2);     # 00000100
const RIGHT       = convert(Action, 1<<3);     # 00001000
const LOAD        = convert(Action, 1<<4);     # 00010000
const UNLOAD      = convert(Action, 1<<5);     # 00100000

const NOTHING     = UP   | DOWN ;              # 00000011
const UP_LEFT     = UP   | LEFT ;              # 00000101
const UP_RIGHT    = UP   | RIGHT;              # 00001001
const DOWN_LEFT   = DOWN | LEFT ;              # 00000110
const DOWN_RIGHT  = DOWN | RIGHT;              # 00001010

const EMPTY       = convert(CellInfo, 0);

const IS_LOADED   = convert(CellInfo, 1<<6);   # 01000000
const IS_BUILDING = convert(CellInfo, 1<<7);   # 10000000

const MINE        = IS_BUILDING | LOAD;        # 10010000
const WAREHOUSE   = IS_BUILDING | UNLOAD;      # 10100000
const WALL        = IS_BUILDING | NOTHING;     # 10000011


const valid_actions =
    Action[NOTHING,
           UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT,
           LOAD, UNLOAD];

dy(action::Action) = action & UP > 0 ? -1 : action & DOWN > 0 ? 1 : 0;
dx(action::Action) = action & LEFT > 0 ? -1 : action & RIGHT > 0 ? 1 : 0;


# Constants used not to screw up localization
const ROW = 1
const COLUMN = 2

# ------------------------------------------------------------------------------

immutable type GlobalParameters
    SEASONS_NO::Integer
    EPISODES_NO::Integer
    STEP::Integer
    EVAL_TRIALS::Integer
    TRIAL_LENGTH::Integer
    RADIUS::Integer
    HEIGHT::Integer
    WIDTH::Integer
    AGENTS_NO::Integer
    WAREHOUSES_NO::Integer
    MINES_NO::Integer
    K::Integer
end

type GlobalState
    board::Array{CellInfo, 2}
    warehouses::Array{Int64, 2}
    mines::Array{Int64, 2}
    agents::Array{Int64, 2}
    last_K::Array{Float64, 2}
    j::Int64
end

function init_scenario(gp::GlobalParameters)
    board = fill(EMPTY, gp.WIDTH, gp.HEIGHT)
    #=
        Warehouses and mines will be placed such that no two
        buildings can be seen in two consecutive states by the same agent.
        That means that the distance between any two buildings must be at least
        2 * RADIUS + 2.

        CURRENT VERSION: 1 MINE + 1 WAREHOUSE
    =#
    if gp.WAREHOUSES_NO > 1 || gp.MINES_NO > 1
        error("Just 1 warehouse and 1 mine... sorry");
    end
    b_rows = shuffle([2 * gp.RADIUS + 1, gp.HEIGHT - 2 * gp.RADIUS]);
    # Mark deposit on board
    warehouses = reshape([b_rows[1],
                          rand((2 * gp.RADIUS + 1):(gp.WIDTH - 2 * gp.RADIUS))],
                         2, gp.WAREHOUSES_NO);
    for d in 1:gp.WAREHOUSES_NO
        board[warehouses[ROW, d], warehouses[COLUMN, d]] = WAREHOUSE;
    end
    last_K = zeros(Float64, gp.K, gp.WAREHOUSES_NO);
    # Mark mines on board
    mines = reshape([b_rows[2],
                     rand((2 * gp.RADIUS + 1):(gp.WIDTH - 2 * gp.RADIUS))],
                    2, gp.MINES_NO);
    for m in 1:gp.MINES_NO
        board[mines[ROW, m], mines[COLUMN, m]] = MINE;
    end
    # Spread agents on board
    agents = zeros(Int64, 2, gp.AGENTS_NO);
    agents[ROW,:] = rand(1:gp.HEIGHT, gp.AGENTS_NO);
    agents[COLUMN,:] = rand(1:gp.WIDTH, gp.AGENTS_NO);
    for i in 1:gp.AGENTS_NO
        while board[agents[ROW, i], agents[COLUMN, i]] != 0
            agents[ROW, i] = rand(1:gp.HEIGHT);
            agents[COLUMN, i] = rand(1:gp.WIDTH);
        end
        board[agents[ROW, i], agents[COLUMN, i]] = valid_actions[rand(1:end)];
    end
    # Put everything together
    return GlobalState(board, warehouses, mines, agents, last_K, 1);
end

function perceive(P::GlobalParameters, S::GlobalState, i::Integer)
    t_idx = S.agents[ROW, i] - P.RADIUS;
    b_idx = S.agents[ROW, i] + P.RADIUS;
    l_idx = S.agents[COLUMN, i] - P.RADIUS;
    r_idx = S.agents[COLUMN, i] + P.RADIUS;

    t_rows = max(0,  1 - t_idx);
    b_rows = max(0, b_idx - P.HEIGHT);
    l_cols = max(0, 1 - l_idx);
    r_cols = max(0, r_idx - P.WIDTH);

    const diameter = 2 * P.RADIUS + 1;
    const mid =  P.RADIUS + 1;

    state =  vcat(fill(WALL, t_rows, diameter),
                  hcat(fill(WALL, diameter - t_rows - b_rows, l_cols),
                       S.board[max(1, t_idx):min(P.HEIGHT, b_idx),
                               max(1, l_idx):min(P.WIDTH, r_idx)],
                       fill(WALL, diameter - t_rows - b_rows, r_cols)),
                  fill(WALL, b_rows, diameter));

    state[mid, mid] = IS_LOADED & state[mid, mid];
    return state;
end

function do_load(S::GlobalState, ag::Integer)
    row = S.agents[ROW, ag];
    col = S.agents[COLUMN, ag];
    if S.board[row, col] & IS_LOADED == 0
        for m in 1:size(S.mines, 2)
            if abs(row - S.mines[ROW, m]) + abs(col - S.mines[COLUMN, m]) == 1
                # The LOAD action succeedes
                S.board[row, col] = IS_LOADED | LOAD;
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
    if S.board[row, col] & IS_LOADED > 0
        for w in 1:size(S.warehouses, 2)
            if (abs(row - S.warehouses[ROW, w]) +
                abs(col - S.warehouses[COLUMN, w])) == 1
                # The UNLOAD action succeeds
                S.board[row, col] = UNLOAD;
                S.last_K[S.j, w] = S.last_K[S.j, w] + 1.0;
                reward = sum(S.last_K[:, w]);
                break;
            end
        end
    end
    return reward;
end

function do_move(P::GlobalParameters, S::GlobalState,
                 ag::Integer, action::Action)
    row = S.agents[ROW, ag];
    col = S.agents[COLUMN, ag];
    if action & NOTHING == NOTHING
        return
    end
    δrows = dy(action)
    δcols = dx(action)
    new_row = min(P.HEIGHT, max(1, row + δrows))
    new_col = min(P.WIDTH, max(1, col + δcols))
    if S.board[new_row, new_col] == 0
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

function log_decay(min_val, max_val, step, steps_no)
    return max_val - log(1+(10-1)*step/steps_no) / log(10) * (max_val-min_val);
end

function update_qs(Qs::Dict{AgentState, Dict{Action, Float64}},
                   s1::AgentState, a::Action, reward::Float64, s2::AgentState,
                   γ::Float64, α::Float64)
    As_Qs = get!(Qs, s1, Dict{Action, Float64}());
    oldQ = get!(As_Qs, a, 0.0);
    As_Qs[a] = (1-α) * oldQ + α * (reward + γ * get_best(Qs, s2)[1]);
    nothing
end

function learn(P::GlobalParameters)
    Qs = Array(Dict{AgentState, Dict{Action, Float64}}, P.AGENTS_NO);
    for i in 1:P.AGENTS_NO
        Qs[i] = Dict{AgentState, Dict{Action, Float64}}();
    end
    #-------------------
    actions = Array(Action, P.AGENTS_NO);
    prev_states = Array(AgentState, P.AGENTS_NO);
    states = Array(AgentState, P.AGENTS_NO);
    rewards = Array(Float64, P.AGENTS_NO);
    #-------------------
    # evaluation
    const evals_no = convert(Integer, P.SEASONS_NO * P.EPISODES_NO / P.STEP);
    avg_score = Array(Float64, evals_no);
    max_score = Array(Float64, evals_no);
    min_score = Array(Float64, evals_no);
    states_no = Array(Float64, evals_no);
    #------------------
    for season in 1:P.SEASONS_NO
        S = init_scenario(P);
        for i in 1:P.AGENTS_NO
            states[i] = perceive(P, S, i);
        end
        max_ϵ = log_decay(0.01, 0.3, season, P.SEASONS_NO);
        j = (season % P.AGENTS_NO) + 1
        # -------------------
        for ep in 1:P.EPISODES_NO
            #= All agents select their actions. =#
            ϵ = log_decay(0.003, max_ϵ, ep, P.EPISODES_NO);
            for ag in 1:P.AGENTS_NO
                if ag == j
                    actions[ag] = choose_action(states[ag], Qs[ag]);
                else
                    actions[ag] = get_best(Qs[ag], states[ag])[2];
                end
                prev_states[ag] = deepcopy(states[ag]);
            end
            #=
                All warehouses put a zero on current time
            =#
            for w in 1:P.WAREHOUSES_NO
                S.last_K[S.j, w] = 0.0;
            end
            #=
                All agents perform their actions and the board changes.
            =#
            for i in shuffle([1:P.AGENTS_NO])
                rewards[i] = 0.0;
                if actions[i] & LOAD > 0
                    do_load(S, i);
                elseif actions[i] & UNLOAD > 0
                    rewards[i] = do_unload(S, i);
                else
                    do_move(P, S, i, actions[i]);
                end
            end
            #=
                Increment j
            =#
            S.j = S.j % P.K + 1;
            #=
                Update Q values
            =#
            for i = 1:P.AGENTS_NO
                states[i] = perceive(P, S, i);
            end

            update_qs(Qs[j], prev_states[j], actions[j], rewards[j], states[j],
                      0.95, 0.05);

            println(season, " x ", ep);
            if ep % P.STEP == 0
                idx = ((season - 1) * P.EPISODES_NO + ep) / P.STEP;
                scores = Array(Float64, P.EVAL_TRIALS);
                for es in 1:P.EVAL_TRIALS
                    S0 = init_scenario(P);
                    scores[es] = evaluate(P, S0, Qs);
                end
                avg_score[idx] = mean(scores);
                max_score[idx] = maximum(scores);
                min_score[idx] = minimum(scores);
                states_no[idx] = mean(map(i->length(keys(Qs[i])),1:P.AGENTS_NO))
            end
        end

        last_idx = season * P.EPISODES_NO / P.STEP;
        df_season_end = DataFrame(x=(1:last_idx),
                                  y=avg_score[1:last_idx],
                                  ymin=min_score[1:last_idx],
                                  ymax=max_score[1:last_idx],
                                  f="delivered goods");
        df_states = DataFrame(x=(1:last_idx), y=states_no[1:last_idx],
                              f="States")
        draw(SVG("results/scores_$season.svg", 6inch, 3inch),
             plot(df_season_end, x=:x, y=:y, ymin=:ymin, ymax=:ymax, color=:f,
                  Geom.line, Geom.ribbon,
                  Guide.xlabel("every $(P.STEP) episodes"),
                  Guide.ylabel("Score"),
                  Guide.title("At the end of season $season")));
        draw(SVG("results/states_$season.svg", 6inch, 3inch),
             plot(df_states, x=:x, y=:y, color=:f, Geom.line,
                  Guide.xlabel("every $(P.STEP) episodes"),
                  Guide.ylabel("avg. # of states"),
                  Guide.title("At the end of season $season")));
        # Save Qs for later use
    end
    JLD.save("results/qs_S$season.jld", "Qs", Qs);
    return Qs;
end


function evaluate(P::GlobalParameters,
                  S::GlobalState,
                  Qs::Array{Dict{AgentState, Dict{Action, Float64}},1})
    #-------------------
    actions = Array(Action, P.AGENTS_NO);
    states = Array(AgentState, P.AGENTS_NO);
    #=
        All warehouses put a zero on current time
    =#
    for d in 1:P.WAREHOUSES_NO
        S.last_K[S.j, d] = 0.0;
    end

    for i in 1:P.AGENTS_NO
        states[i] = perceive(P, S, i);
    end
    # -------------------
    for ep in 1:P.TRIAL_LENGTH
        #= All agents select their actions. =#
        for ag in 1:P.AGENTS_NO
            actions[ag] = get_best(Qs[ag], states[ag])[2];
        end
        #=
            All agents perform their actions and the board changes.
        =#
        for i in shuffle([1:P.AGENTS_NO])
            if actions[i] & LOAD > 0
                do_load(S, i);
            elseif actions[i] & UNLOAD > 0
                do_unload(S, i);
            else
                do_move(P, S, i, actions[i]);
            end
        end
        #=
            Perceive new state
        =#
        for i = 1:P.AGENTS_NO
            states[i] = perceive(P, S, i);
        end
    end
    return sum(S.last_K[S.j, :]);
end


const to_screen = Dict{CellInfo, Char}(NOTHING => char('\u25a1'),
                                       UP => char('\u2191'),
                                       UP_RIGHT => char('\u2197'),
                                       RIGHT => char('\u2192'),
                                       DOWN_RIGHT => char('\u2198'),
                                       DOWN => char('\u2193'),
                                       DOWN_LEFT => char('\u2199'),
                                       LEFT => char('\u2190'),
                                       UP_LEFT => char('\u2196'),
                                       NOTHING | IS_LOADED => char('\u25a0'),
                                       UP | IS_LOADED => char('\u21d1'),
                                       UP_RIGHT | IS_LOADED => char('\u21d7'),
                                       RIGHT | IS_LOADED => char('\u21d2'),
                                       DOWN_RIGHT | IS_LOADED => char('\u21d8'),
                                       DOWN | IS_LOADED => char('\u21d3'),
                                       DOWN_LEFT | IS_LOADED => char('\u21d9'),
                                       LEFT | IS_LOADED => char('\u21d0'),
                                       UP_LEFT | IS_LOADED => char('\u21d6'),
                                       WAREHOUSE => char('\u25c7'),
                                       MINE => char('\u25c6'),
                                       EMPTY => ' ',
                                       LOAD | IS_LOADED => char('\u25a0'),
                                       UNLOAD | IS_LOADED => char('\u25a0'),
                                       LOAD => char('\u25a1'),
                                       UNLOAD => char('\u25a1'),
                                       WALL => 'X');

function display_board(board)
    println("------------");
    for row in 1:size(board,1)
        for col in 1:size(board,2)
            print(to_screen[board[row, col]]);
        end
        println();
    end
    println("------------");
end

const P = GlobalParameters(10^6, 10^4,    # Learning:   SEASONS x EPISODES
                           10^4,        # STEP
                           10, 1000,      # Evaluation: TRIALS x LENGTH
                           2,             # Sensing RADIUS for agents
                           15, 15,        # Board: HEIGHT x WIDTH
                           10, 1, 1, 20); # AGENTS, WAREHOUSES, MINES, K
learn(P);
