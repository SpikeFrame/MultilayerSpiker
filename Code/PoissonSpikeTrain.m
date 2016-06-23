function [ ST ] = PoissonSpikeTrain( Param )
%POISSONSPIKETRAIN Poisson spike train with refractoriness

% Base firing rate
r = Param.R_BAR;
% Refractory period
TAU_R = 10;
COEFF_TRACE_R = Param.DT / TAU_R;

ST = [];
for t_step = 1:Param.N_ITERATIONS
    % Firing rate
    r = r + COEFF_TRACE_R * (Param.R_BAR - r);
    
    if rand < r * Param.DT
        ST(end + 1) = Param.DT * (t_step - 1);
        r = 0;
    end
end

end

