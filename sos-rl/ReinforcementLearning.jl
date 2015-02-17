module ReinforcementLearning

export learn;

using Scenarios;

const EVAL_EVERY       = 200;
const EVAL_SEASONS_NO  = 10;
const EVAL_EPISODES_NO = 1000;

type Evaluation
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

function ϵGreedy{AgentState, Action}(Qs::Dict{AgentState,Dict{Action,Float64}},
                                     validActions::Array{Action,1},
                                     state::AgentState,
                                     ϵ::Float64 = 0.01)
    if rand() < ϵ
        return validActions[rand(1:end)];
    else
        return bestAction(Qs, validActions, state);
    end
end

function updateQs{AgentState,Action}(Qs::Dict{AgentState,Dict{Action,Float64}},
                                     s1::AgentState, a::Action, r::Float64,
                                     s2::AgentState,
                                     γ::Float64, α::Float64)
    AsQs = get!(Qs, s1, Dict{Action, Float64}());
    oldQ = get!(AsQs, a, 0.0);
    AsQs[a] = (1-α) * oldQ + α * (r + γ * bestScore(Qs, s2));
    nothing
end

function learn{AgentState, Action}(s::Scenario{AgentState, Action},
                                   SEASONS_NO,
                                   EPISODES_NO)
    const AGENTS_NO = s.AGENTS_NO;
    const REWARDERS_NO = s.REWARDERS_NO;

    #= Q values =#
    Qs = Array(Dict{AgentState, Dict{Action, Float64}}, AGENTS_NO);
    for ag in 1:AGENTS_NO
        Qs[ag] = Dict{AgentState, Dict{Action, Float64}}();
    end

    #= Actions, States, Rewards =#
    actions = Array(Action, AGENTS_NO);
    prevStates = Array(AgentState, AGENTS_NO);
    states = Array(AgentState, AGENTS_NO);
    #= Evaluation =#
    #=
    eval = Evaluation(zeros(Int64, EVAL_SEASONS_NO, div(SEASONS_NO,EVAL_EVERY)),
                      zeros(Int64, AGENTS_NO, div(SEASONS_NO,EVAL_EVERY)),
                      zeros(Int64, 8, div(SEASONS_NO,EVAL_EVERY)));
    =#
    # ---
    for season in 1:SEASONS_NO
        #println("S$(season)");
        gs = s.init()
        for ag in 1:AGENTS_NO
            states[ag] = s.perceive(gs, ag);
        end
        max_ϵ = log_decay(0.01, 0.2, season, SEASONS_NO);

        total_score = 0.0;
        for episode in 1:EPISODES_NO
            # Choose actions
            ϵ = log_decay(0.003, max_ϵ, episode, EPISODES_NO);
            for ag in 1:AGENTS_NO
                actions[ag] = ϵGreedy(Qs[ag], s.validActions, states[ag], ϵ);
            end
            prevStates = deepcopy(states);
            # Do actions
            rewards, rewarders = s.doActions(gs, actions);
            for ag in 1:AGENTS_NO
                states[ag] = s.perceive(gs, ag, prevStates[ag]);
            end
            #states = map((i,prev)->s.perceive(gs,i,prev), 1:AGENTS_NO, prevStates);
            total_score += sum(rewards);
            for ag in 1:AGENTS_NO
                updateQs(Qs[ag],
                         prevStates[ag], actions[ag], rewards[ag], states[ag],
                         0.97, 0.05);
            end
        end
        #[println(length(keys(Qs[i]))) for i in 1:AGENTS_NO];
        println(total_score);
        #=
        if mod(season, EVAL_EVERY) == 0
            evaluate(Qs, eval, season);
        end
        =#
    end
    Qs
end

end
