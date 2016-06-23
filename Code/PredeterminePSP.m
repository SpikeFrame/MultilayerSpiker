function [ sum_PSP ] = PredeterminePSP( pattern, N_INPUTS, N_HIDDEN, D, Param )
%PREDETERMINEPSP Predetermined PSP evoked at hidden neurons with delays
%sum_PSP(N_h, N_i, t_step)

% Traces
COEFF_TRACE_M = exp(-Param.DT / Param.TAU_M);
COEFF_TRACE_S = exp(-Param.DT / Param.TAU_S);

% Spike train (logical) of input neurons
X_ST_D = false(N_HIDDEN,N_INPUTS,Param.N_ITERATIONS + Param.D_MAX / Param.DT);
for i = 1:N_HIDDEN
    for j = 1:N_INPUTS
        X_ST_D(i,j,round((pattern{j} + D(i,j)) / Param.DT) + 1) = 1;
    end
end

% Initialize PSP
PSP_trace_M = zeros(N_HIDDEN,N_INPUTS);  % Trace: Membrane
PSP_trace_S = zeros(N_HIDDEN,N_INPUTS);  % Trace: Synaptic
sum_PSP = zeros(N_HIDDEN,N_INPUTS,Param.N_ITERATIONS);
%sum_PSP = zeros(N_HIDDEN,N_INPUTS,Param.N_ITERATIONS, 'single'); % Single float: less memory usage but increased simulation time

for t_step = 1:Param.N_ITERATIONS
    % Update traces
    PSP_trace_M = COEFF_TRACE_M * PSP_trace_M;
    PSP_trace_S = COEFF_TRACE_S * PSP_trace_S;
    
    % PSP evoked at hidden neurons due to input neurons
    fired_i = X_ST_D(:,:,t_step); % Index of firing input neurons
    PSP_trace_M(fired_i) = PSP_trace_M(fired_i) + Param.PSP_COEFF;
    PSP_trace_S(fired_i) = PSP_trace_S(fired_i) + Param.PSP_COEFF;
    
    sum_PSP(:,:,t_step) = PSP_trace_M - PSP_trace_S;
end

end
