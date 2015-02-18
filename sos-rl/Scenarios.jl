module Scenarios

export Scenario

immutable Scenario{AgentState, Action}
    AGENTS_NO::Int64
    REWARDERS_NO::Int64
    init::Function
    perceive::Function
    validActions::Array{Action, 1}
    doActions!::Function
    neighbours::Function
end

end
