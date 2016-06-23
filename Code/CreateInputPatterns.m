function [ Input ] = CreateInputPatterns( N_PATTERNS_CLASS, N_INPUTS, Param )
%CREATEINPUTPATTERNS Create input patterns to the network, as Poisson
%distributed spike trains

% Preallocate patterns
Input.pattern = cell(Param.N_PATTERNS,1);

% Class number of each pattern (equal no. of patterns per class)
Input.class_n = ceil((1:Param.N_PATTERNS) / N_PATTERNS_CLASS);

% Generate patterns
for i = 1:Param.N_PATTERNS
    Input.pattern{i} = cell(N_INPUTS,1);
    for j = 1:N_INPUTS
        Input.pattern{i}{j} = PoissonSpikeTrain(Param);
    end
end

end
