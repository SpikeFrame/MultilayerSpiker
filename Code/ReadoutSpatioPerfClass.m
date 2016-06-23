function [ perf_out, class_n_out ] = ReadoutSpatioPerfClass( ST_out, Target, class_n, N_CLASSES, N_OUTPUTS, Param )
%READOUTSPATIOPERFCLASS Readout neuron performance and class number, over multiple readouts, distance found by summing over all responses, NaN class if
%failed to fire or multiple matching classes

if any(cell2mat(ST_out))
    % summed vRD between output spike trains and each target spike train, per
    % class
    dist_c = zeros(N_CLASSES,1);
    for i = 1:N_CLASSES
        for j = 1:N_OUTPUTS
            dist_c(i) = dist_c(i) + VRDist(ST_out{j}, Target(j).pattern{i}, Param.TAU_C);
        end
    end
    
    % Output class: argument of minimum summed vRD
    class_n_out = find(dist_c == min(dist_c));
    
    % Check for unique matching output class
    if length(class_n_out) == 1
        if isequal(class_n_out, class_n)
            perf_out = 100;
        else
            perf_out = 0;
        end
    else
        perf_out = 0;
        class_n_out = NaN;
    end
else
    perf_out = 0;
    class_n_out = NaN;
end

end
