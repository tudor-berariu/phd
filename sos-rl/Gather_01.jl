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


const SEASONS_NO       = 10000;
const EPISODES_NO      = 5000;

const EVAL_EVERY       = 200;
const EVAL_SEASONS_NO  = 10;
const EVAL_EPISODES_NO = 1000;


type Evaluation
    scores::Array{Int64, 2}
    statesNo::Array{Int64, 2}
    drops::Array{Int64, 2}
end

# ---------------------------------------


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
            rewards, warehouseDrops = doActions(gs, actions);
            update_qs(Qs[ag], prevStates[ag],
                      actions[ag], rewards[ag], states[ag],
                      0.97, 0.05);
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


# learn()
