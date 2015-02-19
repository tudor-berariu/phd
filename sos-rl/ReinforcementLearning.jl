module ReinforcementLearning

export learn;

using Scenarios;

using HDF5, JLD
using Cairo;
using DataFrames;
using Gadfly;

const SEASONS_NO = 50001;
const EPISODES_NO = 40000;

const EVAL_EVERY       = 200;
const EVAL_SEASONS_NO  = 25;
const EVAL_EPISODES_NO = 1000;

const EXCHANGE = false;
const LEARNS_ONE = false;

type Results
    bestScore::Float64
    scores::Array{Float64,2}
    statesNo::Array{Int64,2}
    drops::Array{Int64,2}
end

log_decay(min_val, max_val, step, steps_no) =
    max_val - log(1 + (10-1) * step / steps_no) / log(10) * (max_val - min_val);

function bestScore{State,Action}(Qs::Dict{State,Dict{Action,Float64}},
                                 state::State)
    return maximum(values(get(Qs,state,
                              Dict{Action,Float64}([(zero(Action),0.0)]))));
end

function bestAction{State,Action}(Qs::Dict{State,Dict{Action,Float64}},
                                  validActions::Array{Action,1},
                                  state::State)
    bestA = rand(validActions);
    if haskey(Qs, state)
        AsQs = Qs[state];
        bestQ = -Inf;
        for (a, Q) in AsQs
            if Q > bestQ
                bestA = a;
                bestQ = Q;
            end
        end
    end
    return bestA;
end

function bestPair{State,Action}(Qs::Dict{State,Dict{Action,Float64}},
                                validActions::Array{Action,1},
                                state::State)
    bestA = rand(validActions);
    bestQ = 0.0;
    if haskey(Qs, state)
        AsQs = Qs[state];
        for (a, Q) in AsQs
            if Q > bestQ
                bestA = a;
                bestQ = Q;
            end
        end
    end
    return (bestA, bestQ);
end


function ϵGreedy{State, Action}(Qs::Dict{State,Dict{Action,Float64}},
                                validActions::Array{Action,1},
                                state::State,
                                ϵ::Float64)
    if rand() < ϵ
        return validActions[rand(1:end)];
    else
        return bestAction(Qs, validActions, state);
    end
end

function ϵGreedy2{State, Action}(Qs::Dict{State,Dict{Action,Float64}},
                                 validActions::Array{Action, 1},
                                 state::State,
                                 ϵ::Float64,
                                 neighbourA::Action,
                                 neighbourQ::Float64)
    if rand() < ϵ
        return validActions[rand(1:end)];
    else
        A, Q = bestPair(Qs, validActions, state);
        return Q > neighbourQ ? A : neighbourA;
    end
end

function updateQs{State,Action}(Qs::Dict{State,Dict{Action,Float64}},
                                s1::State, a::Action, r::Float64, s2::State,
                                γ::Float64, α::Float64)
    AsQs = get!(Qs, s1, Dict{Action, Float64}());
    oldQ = get!(AsQs, a, 0.0);
    AsQs[a] = (1-α) * oldQ + α * (r + γ * bestScore(Qs, s2));
    nothing
end

function learn{State,Action}(s::Scenario{State,Action})
    const AGENTS_NO = s.AGENTS_NO;
    const REWARDERS_NO = s.REWARDERS_NO;

    #= Q values =#
    Qs = Array(Dict{State, Dict{Action, Float64}}, AGENTS_NO);
    for ag in 1:AGENTS_NO
        Qs[ag] = Dict{State, Dict{Action, Float64}}();
    end

    #= Actions, States, Rewards =#
    actions = Array(Action, AGENTS_NO);
    prevStates = Array(State, AGENTS_NO);
    states = Array(State, AGENTS_NO);
    #= Evaluation =#
    const evals_no = div(SEASONS_NO - 1 ,EVAL_EVERY) + 1;
    results = Results(0.0,
                      zeros(Int64, EVAL_SEASONS_NO, evals_no),
                      zeros(Int64, AGENTS_NO, evals_no),
                      zeros(Int64, 8, evals_no));
    # ---
    for season in 1:SEASONS_NO
        #println("S$(season)");
        gs = s.init()
        for ag in 1:AGENTS_NO
            states[ag] = s.perceive(gs, ag);
        end

        max_ϵ = log_decay(0.05, 0.3, season, SEASONS_NO);
        #total_score = 0.0;
        for episode in 1:EPISODES_NO
            # Choose actions
            ϵ = log_decay(0.003, max_ϵ, episode, EPISODES_NO);
            for ag in 1:AGENTS_NO
                if LEARNS_ONE && (mod(div(season,10), AGENTS_NO) + 1) != ag
                    actions[ag] = bestAction(Qs[ag],s.validActions,states[ag]);
                else
                    if EXCHANGE
                        ns = s.neighbours(gs, ag);
                        neighbourA = rand(s.validActions);
                        neighbourQ = 0.0;
                        for n in ns
                            nA, nQ = bestPair(Qs[n],s.validActions,states[ag]);
                            if nQ > neighbourQ
                                neighbourA = nA;
                                neighbourQ = nQ;
                            end
                        end
                        actions[ag] = ϵGreedy2(Qs[ag], s.validActions,
                                               states[ag], ϵ,
                                               neighbourA, neighbourQ);
                    else
                        actions[ag] = ϵGreedy(Qs[ag], s.validActions,
                                              states[ag], ϵ);
                    end
                end
            end

            # Do actions
            rewards, rewarders = s.doActions!(gs, actions);

            states, prevStates = prevStates, states;
            for ag in 1:AGENTS_NO
                states[ag] = s.perceive(gs, ag, prevStates[ag]);
                updateQs(Qs[ag],
                         prevStates[ag], actions[ag], rewards[ag], states[ag],
                         0.9, 0.04);
            end
        end

        if mod(season-1, EVAL_EVERY) == 0
            assess(s, SEASONS_NO, EPISODES_NO, season, Qs, results);
        end
    end
    nothing
end

function assess{State, Action}(s::Scenario{State, Action},
                               SEASONS_NO,
                               EPISODES_NO,
                               season,
                               Qs::Array{Dict{State, Dict{Action, Float64}}, 1},
                               res::Results)
    const AGENTS_NO = s.AGENTS_NO;
    const REWARDERS_NO = s.REWARDERS_NO;

    const idx = div(season - 1, EVAL_EVERY) + 1;

    #= Number of states =#
    res.statesNo[:, idx] = [length(keys(Qs[ag])) for ag in 1:AGENTS_NO];

    #= Actions, States, Rewards =#
    actions = Array(Action, AGENTS_NO);
    prevStates = Array(State, AGENTS_NO);
    states = Array(State, AGENTS_NO);

    # ---
    for evalSeason in 1:EVAL_SEASONS_NO
        gs = s.init()
        for ag in 1:AGENTS_NO
            states[ag] = s.perceive(gs, ag);
        end

        for episode in 1:EVAL_EPISODES_NO
            # Choose actions
            for ag in 1:AGENTS_NO
                actions[ag] = bestAction(Qs[ag], s.validActions, states[ag]);
            end
            # Do actions
            rewards, rewarders = s.doActions!(gs, actions);

            prevStates, states = states, prevStates;
            for ag in 1:AGENTS_NO
                states[ag] = s.perceive(gs, ag, prevStates[ag]);
            end

            res.scores[evalSeason,idx] += sum(rewards);

            for w in 1:REWARDERS_NO
                if rewarders[w] > 0
                    res.drops[rewarders[w],idx] += 1;
                end
            end
        end
    end
        #= Save plots =#
    # Plot number of states
    @time savePlots(season, idx, res, Qs);
    if sum(res.scores[:,idx]) > res.bestScore
        res.bestScore = sum(res.scores[:,idx]);
        @time JLD.save("results/qs.jld", "Qs", Qs);
    end
    nothing
end

function savePlots{State,Action}(season::Int64, idx::Int64,
                                 res::Results,
                                 Qs::Array{Dict{State,Dict{Action,Float64}},1})
    dfStates = DataFrame(Season=1:EVAL_EVERY:season,
                         Max=1.0.*maximum(res.statesNo[:,1:idx],1)[:],
                         Min=1.0.*minimum(res.statesNo[:,1:idx],1)[:],
                         Mean = mean(res.statesNo[:,1:idx],1)[:]);
    statesPlot = plot(dfStates, x=:Season, y=:Mean, ymax=:Max, ymin=:Min,
                      Geom.line, Geom.ribbon,
                      Guide.ylabel("No. of states"),
                      Guide.title("Evolution of dictionary size"),
                      Guide.xlabel("Learning Season"));
    draw(PDF("results/states.pdf", 6inch, 3inch), statesPlot);
    draw(SVG("results/states.svg", 6inch, 3inch), statesPlot);
    # Plot score
    dfScore = DataFrame(Season = 1:EVAL_EVERY:season,
                        Max = 1.0 .* maximum(res.scores[:,1:idx], 1)[:],
                        Min = 1.0 .* minimum(res.scores[:,1:idx], 1)[:],
                        Mean = mean(res.scores[:,1:idx], 1)[:]);
    scorePlot = plot(dfScore, x=:Season, y=:Mean, ymax=:Max, ymin=:Min,
                     Geom.line, Geom.ribbon,
                     Guide.ylabel("Score"),
                     Guide.title("Evolution of score"),
                     Guide.xlabel("Learning Season"));

    draw(PDF("results/scores.pdf", 6inch, 3inch), scorePlot);
    draw(SVG("results/scores.svg", 6inch, 3inch), scorePlot);

    # Plot drops
    dfDrops=DataFrame(Season=[1 + div(i,8) * EVAL_EVERY for i=0:(idx*8-1)],
                      Drops = [mod(i,8)+1 for i=0:(idx*8-1)],
                      Count = res.drops[1:(idx * 8)])
    #println(dfDrops);
    const m = maximum(sum(res.drops,1));
    const m10 = max(1, div(m, 10));
    dropsPlot = plot(dfDrops, x=:Season, y=:Count, color=:Drops,
                     Geom.bar(position=:stack),
                     Guide.yticks(ticks=[0:m10:m]));
    draw(PDF("results/drops.pdf", 6inch, 3inch), dropsPlot);
    draw(SVG("results/drops.svg", 6inch, 3inch), dropsPlot);
    nothing
end


end
