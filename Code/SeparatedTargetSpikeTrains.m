function [ ST_ref ] = SeparatedTargetSpikeTrains( N_CLASSES, N_SPIKES, Param )
%SEPARATEDTARGETSPIKETRAINS Generate target patterns, with separation between patterns based on vRD

VAN_ROSSUM_DIST_MIN = N_SPIKES / 2; % Minimum inter-pattern vRD separation
T_MIN = 40;                         % Minimum spike times (~4x TAU_M, assuming TAU_M = 10 ms)

ST_ref = [];
failed = 0; % Allow for greater number of target patterns through trial and error

while length(ST_ref) < N_CLASSES
    ST = UniformTargetSpikeTrain(N_SPIKES, T_MIN, Param); % Generate uniformly distributed target spike train
    
    % Check separation between other target patterns
    if ~isempty(ST_ref)
        dist_ref = zeros(length(ST_ref),1);
        for i = 1:length(ST_ref)
            dist_ref(i) = VRDist(ST, ST_ref{i}, Param.TAU_C);
        end
        % Check vRD of ST with others in ST_ref
        if min(dist_ref) < VAN_ROSSUM_DIST_MIN
            failed = failed + 1;
            if failed > 1000
                ST_ref = [];
                failed = 0;
            else
                continue
            end
        end
    end
    
    % Add ST to ST_ref
    ST_ref = [ST_ref; {ST}];
end

end

