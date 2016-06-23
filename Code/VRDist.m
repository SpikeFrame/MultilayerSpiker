function [ van_rossum_dist ] = VRDist( ST_out, ST_ref, TAU_C )
%VRDIST Compute van Rossum distance (exact), taking in both spike trains

% Number of spikes in each spike train
N_out = length(ST_out);
N_ref = length(ST_ref);

% Distance due to (convolved ST_out)^2
d_out = 0;
for i = 1:N_out
    for j = 1:N_out
        d_out = d_out + exp(-(2 * max(ST_out(i), ST_out(j)) - ST_out(i) - ST_out(j)) / TAU_C);
    end
end

% Distance due to (convolved ST_ref)^2
d_ref = 0;
for i = 1:N_ref
    for j = 1:N_ref
        d_ref = d_ref + exp(-(2 * max(ST_ref(i), ST_ref(j)) - ST_ref(i) - ST_ref(j)) / TAU_C);
    end
end

% Distance due to (convolved ST_out * ST_ref)
d_out_ref = 0;
for i = 1:N_out
    for j = 1:N_ref
        d_out_ref = d_out_ref + exp(-(2 * max(ST_out(i), ST_ref(j)) - ST_out(i) - ST_ref(j)) / TAU_C);
    end
end

% van Rossum distance
van_rossum_dist = (d_out + d_ref - 2 * d_out_ref) / 2;

end

