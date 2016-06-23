function [ ST ] = UniformTargetSpikeTrain( N_SPIKES, T_MIN, Param )
%UNIFORMTARGETSPIKETRAIN Uniformally distributed target spike train, with inter-spike
%separation of TAU_C over range [T_MIN, T)

DELTA_ITERATIONS = T_MIN / Param.DT + 1; % Initial time shift
N_ITERATIONS = Param.T / Param.DT;

ST = [] ;
fail = 0; % Allow for greater number of target spikes through trial and error

while length(ST) < N_SPIKES
    s = Param.DT * (round((N_ITERATIONS - DELTA_ITERATIONS) * rand) + DELTA_ITERATIONS - 1);
    
    if ~isempty(ST)
        if min(abs(s - ST)) < Param.TAU_C
            fail = fail + 1;
            if fail > 1000
                ST = [];
                fail = 0;
            end
            continue
        end
    end
    
    ST(end + 1) = s;
end
ST = sort(ST); % Spike train in ascending order

end
