#!/usr/local/bin/julia

using Miners
using Scenarios

function test{State, Action}(s::Scenario{State, Action}, seasons, episodes)
    const L = length(s.validActions);
    const AGENTS_NO = s.AGENTS_NO;
    for season in 1:seasons
        gs = s.init();
        ags = map(i->s.perceive(gs, i), 1:AGENTS_NO);
        for episode in 1:episodes
            actions = [s.validActions[rand(1:L)] for a in 1:AGENTS_NO];
            s.doActions!(gs, actions);
            ags = map((i,ag)->s.perceive(gs, i, ag), 1:AGENTS_NO, ags);
            map(i->s.neighbours(gs, i), 1:AGENTS_NO);
        end
    end
end

test(Gatherers, 10, 20);

@time test(Gatherers, 5000, 1000);
