function Record = Main( N_PATTERNS_CLASS, N_CLASSES, N_SPIKES, N_INPUTS, N_HIDDEN, N_OUTPUTS, N_EPISODES, varargin )
%MAIN Multilayer spiking network (supervised backpropagation) with escape noise
% neurons (soft reset) with axonal conduction delays between input and
% hidden layers
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
Param.ETA_H = 10 / (N_INPUTS * N_SPIKES * N_OUTPUTS);	% Hidden layer
Param.ETA_O = 0.1 / N_HIDDEN;                           % Output layer

% Network
Param.W_H_MAX = 300 / N_INPUTS;	% Max value of initialized hidden weights: <firing_rate> ~ 12 Hz
Param.W_O_MAX = 30 / N_HIDDEN;	% Max value of initialized output weights: <firing_rate> ~ 1 Hz

% Number of input patterns
Param.N_PATTERNS = N_PATTERNS_CLASS * N_CLASSES;

%%%%%%%%%%%%%%%%%%%%%
%%% Setup network %%%
%%%%%%%%%%%%%%%%%%%%%

% Preallocate
W = cell(2,1);          % Weights
E_error = cell(2,1);    % Characteristic eligibility
W_tag = cell(2,1);      % Candidate weight change

%%% Hidden (first) layer connectivity %%%

W{1} = Param.W_H_MAX * rand(N_HIDDEN,N_INPUTS);                         % Uniform distributed weights: W_hi in (0, W_H_MAX]
D = ceil(Param.D_MAX / Param.DT * rand(N_HIDDEN,N_INPUTS)) * Param.DT;  % Conduction delays: D(N_h, N_i) in (0, D_MAX]

%%% Output (second) layer connectivity %%%

W{2} = Param.W_O_MAX * rand(N_OUTPUTS,N_HIDDEN);	% Uniform distributed weights: W_oh in (0, W_O_MAX]

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

for i = 1:N_OUTPUTS
    Target(i) = CreateTargetPatterns(N_CLASSES, N_SPIKES, Param);
end

%%%%%%%%%%%%%%%%%%
%%% Recordings %%%
%%%%%%%%%%%%%%%%%%

%%% Episodic %%%

% Hidden neurons
Record.ST_h = cell(N_EPISODES,N_HIDDEN);          % Hidden spike train
% Record.rate_h = zeros(N_EPISODES,N_HIDDEN);       % Hidden firing rate
% Record.W_h = zeros(N_EPISODES,N_INPUTS,N_HIDDEN); % Input->hidden weight values

% Output neuron
Record.ST_o = cell(N_EPISODES,N_OUTPUTS);
% Record.rate_o = zeros(N_EPISODES,N_OUTPUTS);
% Record.W_o = zeros(N_EPISODES,N_HIDDEN,N_OUTPUTS);

% Pattern classification
Record.vRD = zeros(N_EPISODES,1, 'single');     % vRD per output
Record.perf = zeros(N_EPISODES,1, 'single');	% Performance per output
% Record.class = cell(N_EPISODES,1);
for i = 1:N_OUTPUTS
    Record.ST_ref{i} = Target(i).pattern;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Simulation START %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

for episode = 1:N_EPISODES
    % Print learning epsiode number
    if ~mod(episode,100)
        fprintf('\nEpisode %d\n',episode);
    end
    
    %%% Episodic neuronal initialization %%%
    
    % Hidden layer
    Y = false(N_HIDDEN,Param.N_ITERATIONS); % Logical hidden spike train
    reset_h = zeros(N_HIDDEN,1);            % Reset kernel
    S_trace_M = zeros(N_HIDDEN,N_INPUTS);   % Trace of correlation between input and hidden spike trains (TAU_M component)
    S_trace_S = zeros(N_HIDDEN,N_INPUTS);   % Trace of correlation between input and hidden spike trains (TAU_S component)
    W_tag{1} = zeros(N_HIDDEN,N_INPUTS);    % Candidate weight change: W_hi
    
    % Output layer
    Z = false(N_OUTPUTS,Param.N_ITERATIONS);    % Logical output spike train
    reset_o = zeros(N_OUTPUTS,1);               % Reset kernel
    PSP_o_trace = zeros(2,N_HIDDEN);            % PSP at output neurons due to hidden layer: row 1 = TAU_M component, row 2 = TAU_S
    W_tag{2} = zeros(N_OUTPUTS,N_HIDDEN);       % Candidate weight change: W_oh
    
    %%% Select random input / target output training pattern pair %%%
    
    % Input pattern and associated class
    pattern_n = ceil(Param.N_PATTERNS * rand);
    class_n = Input.class_n(pattern_n);
    
    for t_step = 1:Param.N_ITERATIONS
        %%% Update traces %%%
        
        % Correlated input->hidden neuron firing
        S_trace_M = Param.COEFF_TRACE_M * S_trace_M;
        S_trace_S = Param.COEFF_TRACE_S * S_trace_S;
        
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
        FI_h(FI_h > 1 / Param.DT) = 1 / Param.DT; % Keep probability bound between 0 and 1
        
        % Fired
        Y(:,t_step) = rand(N_HIDDEN,1) < FI_h * Param.DT;
        
        % Trace updates
        reset_h(Y(:,t_step)) = reset_h(Y(:,t_step)) + Param.RESET_K;                % Reset kernel
        PSP_o_trace(:,Y(:,t_step)) = PSP_o_trace(:,Y(:,t_step)) + Param.PSP_COEFF;  % Update PSP trace
        PSP_o = PSP_o_trace(1,:) - PSP_o_trace(2,:);                                % PSP evoked at output layer due to hidden neurons
        
        %%%%%%%%%%%%%%%%%%%%%%%
        %%%% Output neuron %%%%
        %%%%%%%%%%%%%%%%%%%%%%%
        
        % SRM
        u_o = W{2} * PSP_o' + reset_o;
        
        % Firing intensity
        FI_o = Param.EXP_K * exp(Param.EXP_BETA_O * (u_o - Param.U_THETA));
        FI_o(FI_o > 1 / Param.DT) = 1 / Param.DT;
        
        % Fired
        Z(:,t_step) = rand(N_OUTPUTS,1) < FI_o * Param.DT;
        
        % Reset kernel update
        reset_o(Z(:,t_step)) = reset_o(Z(:,t_step)) + Param.RESET_K;
        
        %%%%%%%%%%%%%%%%%%%%%
        %%%% Eligibility %%%%
        %%%%%%%%%%%%%%%%%%%%%
        
        %%% Output neuron %%%
        
        % Backpropagated error signal
        backprop_error = zeros(N_OUTPUTS,1);
        for i = 1:N_OUTPUTS
            backprop_error(i) = Target(i).pattern_logical{class_n}(1,t_step) / Param.DT - FI_o(i); % Learning rate absorbs EXP_BETA_O
        end

        % Characteristic eligibility
        E_error{2} = backprop_error * PSP_o;
        
        %%% Hidden neurons %%%
        
        % Correlated input->hidden firing activity
        if any(Y(:,t_step))
            S_trace_M(Y(:,t_step),:) = S_trace_M(Y(:,t_step),:) + Param.PSP_COEFF * PSP_h{pattern_n}(Y(:,t_step),:,t_step);
            S_trace_S(Y(:,t_step),:) = S_trace_S(Y(:,t_step),:) + Param.PSP_COEFF * PSP_h{pattern_n}(Y(:,t_step),:,t_step);
        end
        S_trace = S_trace_M - S_trace_S;
        
        % Characteristic eligibility
        E_error{1} = bsxfun(@times, S_trace, W{2}' * backprop_error); % Learning rate absorbs EXP_BETA_H
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Candidate weight changes %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Hidden neurons
        W_tag{1} = W_tag{1} + E_error{1};
        
        % Output neuron
        W_tag{2} = W_tag{2} + E_error{2};
    end
    
    %%%%%%%%%%%%%%%%%%%%%%
    %%% Weight updates %%%
    %%%%%%%%%%%%%%%%%%%%%%
    
    %%% Hidden neurons %%%
    
    % Eligibility
    W{1} = W{1} + Param.ETA_H * Param.DT * W_tag{1};
    
    % Synaptic scaling
    W{1} = W{1} + SynapticScaling(sum(Y, 2), W{1}, Param);
    
    % Clipping
    W{1}(W{1} > Param.W_MAX) = Param.W_MAX;
    W{1}(W{1} < -Param.W_MAX) = -Param.W_MAX;
    
    %%% Output neuron %%%
    
    % Eligibility
    W{2} = W{2} + Param.ETA_O * Param.DT * W_tag{2};
    
    % Clipping
    W{2}(W{2} > Param.W_MAX) = Param.W_MAX;
    W{2}(W{2} < -Param.W_MAX) = -Param.W_MAX;
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    %%% Network accuracy %%%
    %%%%%%%%%%%%%%%%%%%%%%%%
    
    % Preallocate
    ST_out = cell(1,N_OUTPUTS);
    vrdist_out = zeros(1,N_OUTPUTS);
    
    % Error
    for i = 1:N_OUTPUTS
        % Spike train
        ST_out{i} = (find(Z(i,:)) - 1) * Param.DT;
        
        % Distance
        vrdist_out(i) = VRDist(ST_out{i}, Target(i).pattern{class_n}, Param.TAU_C);
    end
    
    % Performance
    [perf_out, ~] = ReadoutSpatioPerfClass(ST_out, Target, class_n, N_CLASSES, N_OUTPUTS, Param);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Episodic recordings %%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%% Hidden neurons %%%
    
    % Spike train
    for i = 1:N_HIDDEN
        Record.ST_h{episode,i} = (find(Y(i,:)) - 1) * Param.DT;
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
    Record.ST_o(episode,:) = ST_out;
    
%     % Weights
%     for i = 1:N_OUTPUTS
%         Record.W_o(episode,:,i) = W{2}(i,:);
%     end
%     
%     % Firing rate
%     Record.rate_o(episode,:) = sum(Z, 2)';
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Classification recordings %%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Summed vRD w.r.t. target class
    Record.vRD(episode) = sum(vrdist_out);
    
    % Performance
    Record.perf(episode) = perf_out;
    
%     % Classifications
%     Record.class{episode} = class_n_out;
end

end
