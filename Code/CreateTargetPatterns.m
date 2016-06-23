function [ Target ] = CreateTargetPatterns( N_CLASSES, N_SPIKES, Param )
%CREATETARGETPATTERNS Create target output spike patterns

% Generate target spike trains that are separated by vRD >= N_SPIKES / 2
Target.pattern = SeparatedTargetSpikeTrains(N_CLASSES, N_SPIKES, Param);

% Logical target spike trains
Target.pattern_logical = cell(N_CLASSES,1);
for i = 1:N_CLASSES
    Target.pattern_logical{i} = false(1,Param.N_ITERATIONS);
    Target.pattern_logical{i}(round(Target.pattern{i} / Param.DT) + 1) = 1;
end

end

