using Gadfly
using Color
using DataFrames

Season = [div(j-1,8)+1 for j in 1:160];
Drops = [mod(j-1,8)+1 for j in 1:160];
Count = rand(1:20, 160) .+ 30;
df = DataFrame(Season=Season, Drops=Drops, Count=Count);

# Diferite variante

plot(df, x=:Season, y=:Count, color=:Drops, Geom.bar(position=:stack), Guide.yticks(ticks=[0:50:400]))

#=
plot(df, x=:Season, y=:Count, color=:Drops, Geom.bar(position=:stack), Guide.yticks(ticks=[0:50:400]), Scale.ContinuousColorScale(p -> RGB(0, 0, 0.2+0.7*(1-p))))
=#
#=
plot(df, x=:Season, y=:Count, color=:Drops, Geom.bar(position=:stack), Guide.yticks(ticks=[0:50:400]), Scale.ContinuousColorScale(p -> RGB(0.2+ 0.7*(1-p),0.2+0.7*p,0.2+0.7*(1-p))))
=#
#=
plot(df, x=:Season, y=:Count, color=:Drops, Geom.bar(position=:stack), Guide.yticks(ticks=[0:50:400]), Scale.ContinuousColorScale(p -> RGB(0.2+ 0.7*(1-p),0.2+0.7*p,0.2+0.7*(1-p))))
=#
