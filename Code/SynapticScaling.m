function [ Delta_W ] = SynapticScaling( R_out, W, Param )
%SCALINGSCALING Synaptic scaling to maintain homeostatic firing rate

Delta_W = zeros(size(W));

index_MAX = R_out > Param.R_MAX; % Indices for high firing rate
index_MIN = R_out < Param.R_MIN; % Indices for low firing rate

% Excessive activity
Delta_W(index_MAX,:) = bsxfun(@times, Param.SCALING_STRENGTH * (Param.R_MAX - R_out(index_MAX)), abs(W(index_MAX,:)));

% Minimal activity
Delta_W(index_MIN,:) = bsxfun(@times, Param.SCALING_STRENGTH * (Param.R_MIN - R_out(index_MIN)), abs(W(index_MIN,:)));

end

