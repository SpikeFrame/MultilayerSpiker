function Record = MainSingle( N_PATTERNS_CLASS, N_CLASSES, N_SPIKES, N_INPUTS, N_HIDDEN, N_EPISODES, varargin )
%MAINSINGLE Multilayer spiking network (supervised backpropagation) with escape noise
% neurons (soft reset) with axonal conduction delays between input and
% hidden layers
% Optimised for a single output neuron
% This code approximates EXP_BETA_O -> infinity for faster runtime
% Brian Gardner
% 02/05/16

%%% Set stream for RNG %%%
if (~isempty(varargin))
    RandStream.setGlobalStream(varargin{1});
else
    RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));
end

%%%%%%%%%%%%%%%%%%%%%
%%%%  Parameters %%%%
%%%%%%%%%%%%%%%%%%%%%

% Network constants
Param = CreateParam();

% Learning rate
Param.ETA_H = 10 / (N_INPUTS * N_SPIKES);	% Hidden layer
Param.ETA_O = 0.1 / N_HIDDEN;               % Output layer

% Network
Param.W_H_MAX = 300 / N_INPUTS;	% Max value of initialized hidden weights: <firing_rate> ~ 12 Hz
Param.W_O = 12 / N_HIDDEN;      % Same value of initialized output weights: <firing_rate> ~ 1 Hz

% Number of input patterns
Param.N_PATTERNS = N_PATTERNS_CLASS * N_CLASSES;

%%%%%%%%%%%%%%%%%%%%%
%%% Setup network %%%
%%%%%%%%%%%%%%%%%%%%%

% Preallocate
W = cell(2,1);	% Weights

%%% Hidden (first) layer connectivity %%%

W{1} = Param.W_H_MAX * rand(N_HIDDEN,N_INPUTS);                         % Uniform distributed weights: W_hi in (0, W_H_MAX]
D = ceil(Param.D_MAX / Param.DT * rand(N_HIDDEN,N_INPUTS)) * Param.DT;  % Conduction delays: D(N_h, N_i) in (0, D_MAX]

%%% Output (second) layer connectivity %%%

W{2} = Param.W_O * ones(1,N_HIDDEN);  % Same initial weights

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Setup input/target patterns %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Create input patterns %%%

Input = CreateInputPatterns(N_PATTERNS_CLASS, N_INPUTS, Param);

% Predetermined PSP evoked at hidden layer neurons (PSP_h) due to input patterns
PSP_h = cell(Param.N_PATTERNS,1);
for i = 1:Param.N_PATTERNS
    PSP_h{i} = PredeterminePSP(Input.pattern{i}, N_INPUTS, N_HIDDEN, D, Param);
end

%%% Create target output patterns %%%

Target = CreateTargetPatterns(N_CLASSES, N_SPIKES, Param);

%%%%%%%%%%%%%%%%%%
%%% Recordings %%%
%%%%%%%%%%%%%%%%%%

%%% Episodic %%%

% Hidden neurons
Record.ST_h = cell(N_EPISODES,N_HIDDEN);          % Hidden spike train
% Record.rate_h = zeros(N_EPISODES,N_HIDDEN);       % Hidden firing rate
% Record.W_h = zeros(N_EPISODES,N_INPUTS,N_HIDDEN); % Input->hidden weight values

% Output neuron
Record.ST_o = cell(N_EPISODES,1);
% Record.rate_o = zeros(N_EPISODES,1);
% Record.W_o = zeros(N_EPISODES,N_HIDDEN);

% Pattern classification
Record.vRD = zeros(N_EPISODES,1, 'single');     % vRD per output
Record.perf = zeros(N_EPISODES,1, 'single');	% Performance per output
% Record.class = cell(N_EPISODES,1);
Record.ST_ref = Target.pattern;

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Simulation START %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

for episode = 1:N_EPISODES
    % Print learning epsiode number
    if ~mod(episode,1000)
        fprintf('\nEpisode %d\n', episode);
    end
    
    %%% Episodic neuronal initialization %%%
    
    % Hidden layer
    Y = false(N_HIDDEN,Param.N_ITERATIONS); % Logical hidden spike train
    reset_h = zeros(N_HIDDEN,1);            % Reset kernel
    
    % Output layer
    Z = false(1,Param.N_ITERATIONS);            % Logical output spike train
    reset_o = 0;                                % Reset kernel
    PSP_o_trace = zeros(2,N_HIDDEN);            % PSP at output neurons due to hidden layer: row 1 = TAU_M component, row 2 = TAU_S
    PSP_o = zeros(Param.N_ITERATIONS,N_HIDDEN);
    
    %%% Select random input / target output training pattern pair %%%
    
    % Input pattern and associated class
    pattern_n = ceil(Param.N_PATTERNS * rand);
    class_n = Input.class_n(pattern_n);
    
    for t_step = 1:Param.N_ITERATIONS
        %%% Update traces %%%
        
        % Hidden layer neurons
        reset_h = Param.COEFF_TRACE_M * reset_h;
        
        % Output layer neurons
        reset_o = Param.COEFF_TRACE_M * reset_o;
        PSP_o_trace(1,:) = Param.COEFF_TRACE_M * PSP_o_trace(1,:);
        PSP_o_trace(2,:) = Param.COEFF_TRACE_S * PSP_o_trace(2,:);
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        %%%% Hidden neurons %%%%
        %%%%%%%%%%%%%%%%%%%%%%%%
        
        % Simplified Spike Response Model (SRM)
        u_h = sum(W{1} .* PSP_h{pattern_n}(:,:,t_step), 2) + reset_h;
        
        % Firing intensity
        FI_h = Param.EXP_K * exp(Param.EXP_BETA_H * (u_h - Param.U_THETA));
        FI_h(FI_h > 1 / Param.DT) = 1 / Param.DT;   % Keep probability bound between 0 and 1
        
        % Fired
        Y(:,t_step) = rand(N_HIDDEN,1) < FI_h * Param.DT;
        
        % Trace updates
        reset_h(Y(:,t_step)) = reset_h(Y(:,t_step)) + Param.RESET_K;                % Reset kernel
        PSP_o_trace(:,Y(:,t_step)) = PSP_o_trace(:,Y(:,t_step)) + Param.PSP_COEFF;  % Update PSP trace
        PSP_o(t_step,:) = PSP_o_trace(1,:) - PSP_o_trace(2,:);                      % PSP evoked at output layer due to hidden neurons
        
        %%%%%%%%%%%%%%%%%%%%%%%
        %%%% Output neuron %%%%
        %%%%%%%%%%%%%%%%%%%%%%%
        
        % SRM
        u_o = sum(W{2} .* PSP_o(t_step,:), 2) + reset_o;
        
        % Fired
        if u_o > Param.U_THETA
            Z(t_step) = 1;
            reset_o = reset_o + Param.RESET_K;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    %%% Network accuracy %%%
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    % Readout spike train
    ST_out = (find(Z) - 1) * Param.DT;
    
    % Error
    vrdist_out = VRDist(ST_out, Target.pattern{class_n}, Param.TAU_C);
    
    % Performance
    [perf_out, ~] = ReadoutSpatioPerfClass({ST_out}, Target, class_n, N_CLASSES, 1, Param);
    
    %%%%%%%%%%%%%%%%%%%%%%
    %%% Weight updates %%%
    %%%%%%%%%%%%%%%%%%%%%%
    
    %%% Hidden neurons %%%
    
    ST_h = cell(1,N_HIDDEN);
    t_step_h = cell(1,N_HIDDEN);
    for i = 1:N_HIDDEN
        ST_h{i} = (find(Y(i,:)) - 1) * Param.DT;
        t_step_h{i} = ST_h{i} / Param.DT + 1;
    end
    
    % Weight change
    dW = zeros(N_HIDDEN,N_INPUTS);
    % Potentiation
    for i = 1:length(Target.pattern{class_n})
        for j = 1:N_HIDDEN
            for k = 1:length(ST_h{j})
                if ST_h{j}(k) < Target.pattern{class_n}(i)
                    s = ST_h{j}(k) - Target.pattern{class_n}(i);
                    dW(j,:) = dW(j,:) + Param.PSP_COEFF * (exp(s / Param.TAU_M) - exp(s / Param.TAU_S)) * PSP_h{pattern_n}(j,:,t_step_h{j}(k));
                else
                    break
                end
            end
        end
    end
    % Depression
    for i = 1:length(ST_out)
        for j = 1:N_HIDDEN
            for k = 1:length(ST_h{j})
                if ST_h{j}(k) < ST_out(i)
                    s = ST_h{j}(k) - ST_out(i);
                    dW(j,:) = dW(j,:) - Param.PSP_COEFF * (exp(s / Param.TAU_M) - exp(s / Param.TAU_S)) * PSP_h{pattern_n}(j,:,t_step_h{j}(k));
                else
                    break
                end
            end
        end
    end
    W{1} = W{1} + Param.ETA_H * bsxfun(@times, W{2}', dW);
    
    % Synaptic scaling
    W{1} = W{1} + SynapticScaling(sum(Y, 2), W{1}, Param);
    
    % Clipping
    W{1}(W{1} > Param.W_MAX) = Param.W_MAX;
    W{1}(W{1} < -Param.W_MAX) = -Param.W_MAX;
    
    %%% Output neuron %%%
    
    % Weight chage
    dW = zeros(1,N_HIDDEN);
    % Potentiation
    for i = 1:length(Target.pattern{class_n})
        t_step_o_ref = Target.pattern{class_n}(i) / Param.DT + 1;
        dW = dW + PSP_o(t_step_o_ref,:);
    end
    % Depression
    for i = 1:length(ST_out)
        t_step_o = ST_out(i) / Param.DT + 1;
        dW = dW - PSP_o(t_step_o,:);
    end
    W{2} = W{2} + Param.ETA_O * dW;
    
    % Clipping
    W{2}(W{2} > Param.W_MAX) = Param.W_MAX;
    W{2}(W{2} < 0.01) = 0.01;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Episodic recordings %%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Hidden neurons %%%
    
    % Spike train
    for i = 1:N_HIDDEN
        Record.ST_h{episode,i} = (find(Y(1,:)) - 1) * Param.DT;
    end
    
%     % Weights
%     for i = 1:N_HIDDEN
%         Record.W_h(episode,:,i) = W{1}(i,:);
%     end
%     
%     % Firing rate
%     Record.rate_h(episode,:) = sum(Y, 2)';
    
    %%% Output neuron %%%
    
    % Spike train
    Record.ST_o{episode,1} = ST_out;
    
%     % Weights
%     Record.W_o(episode,:) = W{2};
%     
%     % Firing rate
%     Record.rate_o(episode,1) = sum(Z);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Classification recordings %%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % vRD w.r.t. target class
    Record.vRD(episode) = vrdist_out;
    
    % Performance
    Record.perf(episode) = perf_out;
    
%     % Classifications
%     Record.class{episode} = class_n_out;
end

end
