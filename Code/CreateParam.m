function [ Param ] = CreateParam( )
%CREATEPARAM Create network parameters

% Time / patterns
Param.DT = 1;                               % Time step [ms]
Param.T = 500;                              % Pattern duration / time period [ms]
Param.TAU_C = 10;                           % van Rossum Distance time constant [ms]
Param.N_ITERATIONS = Param.T / Param.DT;    % Total number of time steps per episode: t_step in [0, T)
Param.R_BAR = 6 / 1000;                     % Mean firing rate per input neuron [1 / ms]

% Neurons
Param.U_THETA = 15;                         % Firing threshold (mV)
Param.RESET_K = -15;                        % Reset kernel
Param.D_MAX = 40;                           % Max conduction delay

% Escape rate
Param.EXP_K = 0.01;                         % Firing rate at threshold [1 / ms]
Param.EXP_BETA_H = 1 / 2;                   % Hidden neurons inverse width of firing threshold [1 / mV]
Param.EXP_BETA_O = 5;                       % Output neurons inverse width of firing threshold [1 / mV]

% PSP
Param.TAU_M = 10;                           % Membrane time constant [ms]
Param.TAU_S = 5;                            % Synaptic time constant [ms]
Param.PSP_COEFF = 4;                        % PSP coefficient [mV]

% Traces
Param.COEFF_TRACE_M = exp(-Param.DT / Param.TAU_M);     % Membrane trace decay
Param.COEFF_TRACE_S = exp(-Param.DT / Param.TAU_S);     % Synaptic trace decay

% Synaptic scaling
Param.R_MAX = 20;                           % Maximum firing rate [spikes / period]
Param.R_MIN = 1;                            % Minimum firing rate [spikes / period]
Param.SCALING_STRENGTH = 1e-2;              

% Clipping
Param.W_MAX = 100;

end

