#!/usr/local/bin/julia

using Miners

function test(seasons, episodes)
    const L = length(validActions);
    for season in 1:seasons
        gs = initialState();
        ags = map(i->perceiveMap(gs, i), 1:AGENTS_NO);
        for episode in 1:episodes
            actions = [validActions[rand(1:L)] for a in 1:AGENTS_NO];
            doActions(gs, actions);
            ags = map((i,ag)->perceiveMap(gs, i, ag), 1:AGENTS_NO, ags);
        end
    end
end

test(10, 20);

@time test(50, 1000);
