function [out, index] = hadmm_solver(kmax, k, z, K, factor, M, N, varargin)
%HADMM_SOLVER MUSIC-guided hierarchical local solver.
% Clean numerical path: no timers, history buffers or diagnostic residuals.
% Read: candidate pool -> refine_candidate -> run_stage_search -> local_refine.
% Instrumented twin: hadmm_solver_diagnostic.m. Numerical rules are identical.

    fast_path_enabled = true;
    zoom_rounds = 5;
    local_solver = 'admm_s';
    admm_iterations = kmax;
    enable_subarray_expansion = true;
    enable_residual_augmentation = true;
    enable_candidate_pruning = true;
    joint_refinement_mode = 'gated';
    joint_mode_explicit = false;
    local_refinement_mode = 'grid';
    forced_branch_paths = [];

    if numel(varargin) >= 1 && isstruct(varargin{end})
        opts = varargin{end};
        varargin(end) = [];
        if isfield(opts, 'fast_path_enabled')
            fast_path_enabled = logical(opts.fast_path_enabled);
        end
        if isfield(opts, 'zoom_rounds')
            zoom_rounds = max(1, round(double(opts.zoom_rounds)));
        end
        if isfield(opts, 'local_solver')
            local_solver = lower(char(opts.local_solver));
        end
        if isfield(opts, 'admm_iterations')
            admm_iterations = max(1, round(double(opts.admm_iterations)));
        end
        if isfield(opts, 'enable_subarray_expansion')
            enable_subarray_expansion = logical(opts.enable_subarray_expansion);
        end
        if isfield(opts, 'enable_residual_augmentation')
            enable_residual_augmentation = logical(opts.enable_residual_augmentation);
        end
        if isfield(opts, 'enable_candidate_pruning')
            enable_candidate_pruning = logical(opts.enable_candidate_pruning);
        end
        if isfield(opts, 'joint_refinement_mode')
            joint_refinement_mode = lower(char(opts.joint_refinement_mode));
            joint_mode_explicit = true;
        end
        if isfield(opts, 'local_refinement_mode')
            local_refinement_mode = lower(char(opts.local_refinement_mode));
        end
        if isfield(opts, 'forced_branch_paths') && ...
                ~isempty(opts.forced_branch_paths)
            forced_branch_paths = double(opts.forced_branch_paths);
            assert(ismatrix(forced_branch_paths), ...
                'forced_branch_paths must be a numeric matrix.');
        end
    end

    valid_local_solvers = {'matched_filter', 'ridge', 'admm_s', 'admm_z'};
    if ~any(strcmp(local_solver, valid_local_solvers))
        error('Unsupported local_solver: %s', local_solver);
    end
    valid_joint_modes = {'gated', 'always', 'never'};
    if ~any(strcmp(joint_refinement_mode, valid_joint_modes))
        error('Unsupported joint_refinement_mode: %s', joint_refinement_mode);
    end
    if ~joint_mode_explicit && ~fast_path_enabled
        joint_refinement_mode = 'always';
    end
    valid_refinement_modes = {'grid', 'vectorized_grid'};
    if ~any(strcmp(local_refinement_mode, valid_refinement_modes))
        error('Unsupported local_refinement_mode: %s', ...
            local_refinement_mode);
    end

    if numel(varargin) == 5
        NeighborSearcher = varargin{1};
        threshold = varargin{2};
        lambda = varargin{3};
        rho = varargin{4};
        x = varargin{5};
        music_N = [];
        source_count = 1;
    elseif numel(varargin) == 6
        music_N = varargin{1};
        NeighborSearcher = varargin{2};
        threshold = varargin{3};
        lambda = varargin{4};
        rho = varargin{5};
        x = varargin{6};
        source_count = 1;
    elseif numel(varargin) == 7
        music_N = varargin{1};
        NeighborSearcher = varargin{2};
        threshold = varargin{3};
        lambda = varargin{4};
        rho = varargin{5};
        x = varargin{6};
        source_count = varargin{7};
    else
        error('hadmm_solver received an unsupported number of inputs.');
    end

    if isempty(source_count)
        source_count = 1;
    end

    step = length(factor);
    tau = lambda / rho;

    if isvector(x)
        x_matrix = x(:);
    else
        x_matrix = x;
    end

    validScIndex = 1:min(size(x_matrix, 1), numel(z));
    x1 = x_matrix(validScIndex, :);
    z1 = z(validScIndex);
    sample_count = size(x1, 1);

    if isempty(music_N)
        music_N = min(8, min(numel(z1), sample_count));
    end

    stage_peak_counts = normalize_stage_counts(M, step);
    stage_sensor_counts = normalize_stage_counts(N, step);

    %% 1. Coarse MUSIC candidates and top-level selection
    first_resolution = factor(1);
    phi_list_first = linspace(-pi/2, pi/2, first_resolution)';
    music_sensor_count = min([max(1, music_N), sample_count, numel(z1)]);

    % Explicit inputs for the ordinary local helpers below.
    p = struct;
    p.enable_candidate_pruning = enable_candidate_pruning;
    p.enable_residual_augmentation = enable_residual_augmentation;
    p.enable_subarray_expansion = enable_subarray_expansion;
    p.fast_path_enabled = fast_path_enabled;
    p.first_resolution = first_resolution;
    p.joint_refinement_mode = joint_refinement_mode;
    p.k = k;
    p.local_refinement_mode = local_refinement_mode;
    p.music_sensor_count = music_sensor_count;
    p.phi_list_first = phi_list_first;
    p.sample_count = sample_count;
    p.source_count = source_count;
    p.stage_sensor_counts = stage_sensor_counts;
    p.step = step;
    p.x1 = x1;
    p.z1 = z1;
    p.zoom_rounds = zoom_rounds;

    %% MUSIC candidate pool
    candidate_indices = build_candidate_pool(p);
    candidate_count = numel(candidate_indices);

    candidate_paths = zeros(candidate_count, step);
    candidate_angles = zeros(candidate_count, 1);
    candidate_residuals = inf(candidate_count, 1);
    candidate_grid_spacings = zeros(candidate_count, 1);

    %% Hierarchical windows, ADMM updates and terminal zoom
    for cand = 1:candidate_count
        path_idx = zeros(1, step);
        path_idx(1) = candidate_indices(cand);

        resolution = first_resolution;
        phi_last = phi_list_first;

        for stage = 2:step
            resolution = resolution * factor(stage);

            window_parent_idx = path_idx(stage - 1);
            if ~isempty(forced_branch_paths) && ...
                    cand <= size(forced_branch_paths, 1) && ...
                    stage - 1 <= size(forced_branch_paths, 2)
                forced_parent_idx = forced_branch_paths(cand, stage - 1);
                if isfinite(forced_parent_idx) && forced_parent_idx >= 1
                    window_parent_idx = round(forced_parent_idx);
                end
            end

            upper_bound = window_parent_idx + NeighborSearcher(stage - 1);
            if upper_bound > resolution / factor(stage)
                upper_bound = resolution / factor(stage);
            end

            lower_bound = window_parent_idx - NeighborSearcher(stage - 1);
            if lower_bound < 0
                lower_bound = 0;
            end

            phi_last = linspace(-pi/2, pi/2, resolution)';
            sensor_count = min(stage_sensor_counts(stage), sample_count);
            z_stage = z1(1:sensor_count);
            y_stage = x1(1:sensor_count, :);
            col_start = factor(stage) * lower_bound + 1;
            col_end = factor(stage) * upper_bound;
            local_indices = col_start:col_end;
            S1_mini = exp(1j * k * z_stage * sin(phi_last(local_indices).'));
            anchor_global_idx = factor(stage) * window_parent_idx;
            anchor_local_idx = anchor_global_idx - col_start + 1;

            K_local = size(S1_mini, 2);
            snapshot_count = size(y_stage, 2);

            if strcmp(local_solver, 'matched_filter')
                ranking_state = S1_mini' * y_stage;
            else
                system_matrix = 2 .* (S1_mini' * S1_mini) + ...
                    rho .* eye(K_local);
                a = 2 .* S1_mini' * y_stage;

                B = inv(system_matrix);

                if strcmp(local_solver, 'ridge')
                    ranking_state = B * a;
                else
                    sk = zeros(K_local, snapshot_count);
                    zk = zeros(K_local, snapshot_count);
                    uk = zeros(K_local, snapshot_count);
                    for iter = 1:admm_iterations
                        b = a + rho .* (zk - uk);
                        sk1 = B * b;
                        temp = sk1 + uk;
                        zk1 = shrink_complex(temp, tau);
                        uk1 = uk + sk1 - zk1;
                        sk = sk1;
                        zk = zk1;
                        uk = uk1;
                    end
                    if strcmp(local_solver, 'admm_z')
                        ranking_state = zk;
                    else
                        ranking_state = sk;
                    end
                end
            end

            magnitude = sqrt(sum(abs(ranking_state) .^ 2, 2));
            max_magnitude = max(magnitude);
            if max_magnitude > 0
                sk_stem = magnitude / max_magnitude;
            else
                sk_stem = magnitude;
            end

            selected_idx = select_ranked_peak( ...
                sk_stem, stage_peak_counts(stage), threshold(stage), anchor_local_idx);

            local_idx = selected_idx;

            path_idx(stage) = local_indices(local_idx);
        end

        final_phi = phi_last(path_idx(step));
        if numel(phi_last) > 1
            grid_spacing = phi_last(2) - phi_last(1);
        else
            grid_spacing = 0;
        end
        final_grid_spacing = grid_spacing;

        refined_phi = local_refine(final_phi, grid_spacing, p);
        angle_deg = refined_phi * 180 / pi;
        residual = compute_residual(refined_phi, p);
        candidate_paths(cand, :) = path_idx;
        candidate_angles(cand) = angle_deg;
        candidate_residuals(cand) = residual;
        candidate_grid_spacings(cand) = final_grid_spacing;

    end

    %% Final source-set selection
    if source_count == 1
        [~, best_idx] = min(candidate_residuals);
        out = candidate_angles(best_idx);
        index = candidate_paths(best_idx, :);
    else
        all_angles = candidate_angles;
        all_paths = candidate_paths;
        all_grid_spacings = candidate_grid_spacings;
        % One-pass selection block: an early break accepts the current source set.
        while true
            all_angles = all_angles(:);
            subset_count = min(source_count, numel(all_angles));

            if numel(all_angles) == subset_count
                selected_angles = all_angles(:).';
                selected_paths = all_paths;
                [selected_angles, sort_idx] = sort(selected_angles, 'ascend');
                selected_paths = selected_paths(sort_idx, :);
                selected_grid_spacings = all_grid_spacings(sort_idx);
                if should_accept_without_joint(selected_angles, selected_grid_spacings, p)
                    break;
                end
            end

            combinations = nchoosek(1:numel(all_angles), subset_count);
            min_separation_deg = 1.5 * max(all_grid_spacings) * 180 / pi;
            best_residual = inf;
            best_subset = combinations(1, :);

            for comb_idx = 1:size(combinations, 1)
                current_subset = combinations(comb_idx, :);
                current_angles = sort(all_angles(current_subset));
                if subset_count > 1 && any(diff(current_angles) < min_separation_deg)
                    continue;
                end

                current_residual = compute_joint_residual(current_angles * pi / 180, ...
                    z1(1:min(stage_sensor_counts(step), sample_count)), ...
                    x1(1:min(stage_sensor_counts(step), sample_count), :), p);
                if current_residual < best_residual
                    best_residual = current_residual;
                    best_subset = current_subset;
                end
            end

            selected_angles = all_angles(best_subset);
            selected_paths = all_paths(best_subset, :);
            selected_grid_spacings = all_grid_spacings(best_subset);
            [selected_angles, sort_idx] = sort(selected_angles, 'ascend');
            selected_paths = selected_paths(sort_idx, :);
            selected_grid_spacings = selected_grid_spacings(sort_idx);

            if should_accept_without_joint(selected_angles, selected_grid_spacings, p)
                selected_angles = selected_angles(:).';
                break;
            end

            refined_angles = selected_angles(:);
            refined_paths = selected_paths;
            subset_count = numel(refined_angles);
            refined_residuals = inf(subset_count, 1);

            final_sensor_count = min(stage_sensor_counts(step), sample_count);
            z_joint = z1(1:final_sensor_count);
            y_joint = x1(1:final_sensor_count, :);

            angle_rad_list = refined_angles * pi / 180;
            joint_spacing = max(selected_grid_spacings);
            if joint_spacing <= 0
                joint_spacing = pi / max(prod(factor), 1);
            end

            search_radius = 3 * joint_spacing;
            search_count = 11;
            phi_candidates = cell(subset_count, 1);
            for ii = 1:subset_count
                phi_min = max(-pi / 2, angle_rad_list(ii) - search_radius);
                phi_max = min(pi / 2, angle_rad_list(ii) + search_radius);
                phi_candidates{ii} = linspace(phi_min, phi_max, search_count);
            end

            best_residual = inf;
            best_pair = angle_rad_list;

            if subset_count == 2
                left_candidates = phi_candidates{1};
                right_candidates = phi_candidates{2};
                for ii = 1:numel(left_candidates)
                    phi_left = left_candidates(ii);
                    for jj = 1:numel(right_candidates)
                        phi_right = right_candidates(jj);
                        if phi_right <= phi_left + joint_spacing
                            continue;
                        end
                        residual = compute_joint_residual([phi_left; phi_right], z_joint, y_joint, p);
                        if residual < best_residual
                            best_residual = residual;
                            best_pair = [phi_left; phi_right];
                        end
                    end
                end
            else
                best_residual = compute_joint_residual(angle_rad_list, z_joint, y_joint, p);
            end

            refined_angles = best_pair * 180 / pi;
            refined_residuals = zeros(subset_count, 1);
            final_resolution = prod(factor);
            final_phi_list = linspace(-pi / 2, pi / 2, final_resolution)';

            % Apply per-source 5-round zoom to each source after joint search.
            % The 11x11 grid locates the joint minimum to within ~0.1 deg;
            % local_refine then zooms to the ~2e-6 deg floor (same as single-source).
            for ii = 1:subset_count
                best_pair(ii) = local_refine(best_pair(ii), joint_spacing, p);
            end
            refined_angles = best_pair * 180 / pi;

            for ii = 1:subset_count
                refined_residuals(ii) = compute_residual(best_pair(ii), p);
                [~, nearest_idx] = min(abs(final_phi_list - best_pair(ii)));
                refined_paths(ii, step) = nearest_idx;
            end
            selected_angles = refined_angles;
            selected_paths = refined_paths;

            selected_angles = selected_angles(:).';
            break;
        end
        out = selected_angles;
        index = selected_paths;

    end

end

% Ordinary local functions: explicit input parameters.
function [candidate_indices] = build_candidate_pool(p)
    base_pool = build_candidate_pool_for_sensor_count(p.music_sensor_count, p);
    candidate_indices = base_pool;

    if p.source_count == 1 || is_candidate_pool_confident(base_pool, p.music_sensor_count, p) || ...
            ~p.enable_subarray_expansion
        return;
    end

    sensor_schedule = unique([ ...
        p.music_sensor_count, ...
        min([p.sample_count, numel(p.z1), p.stage_sensor_counts(1), max(p.music_sensor_count + 8, 24)]), ...
        min([p.sample_count, numel(p.z1), p.stage_sensor_counts(1), max(p.music_sensor_count + 16, 32)]) ...
    ]);
    sensor_schedule = sensor_schedule(sensor_schedule > p.music_sensor_count);

    for schedule_idx = 1:numel(sensor_schedule)
        expanded_music_sensor_count = sensor_schedule(schedule_idx);
        expanded_pool = build_candidate_pool_for_sensor_count(expanded_music_sensor_count, p);
        if is_candidate_pool_confident(expanded_pool, expanded_music_sensor_count, p)
            candidate_indices = expanded_pool;
            return;
        end
        candidate_indices = prune_candidate_pool([expanded_pool, candidate_indices], expanded_music_sensor_count, p);
    end
end

function [candidate_indices] = build_candidate_pool_for_sensor_count(sensor_count, p)
    z_music = p.z1(1:sensor_count);
    x_music = p.x1(1:sensor_count, :);

    [~, initial_music_idx] = run_coarse_music(sensor_count, p.source_count, x_music, p);

    candidate_indices = unique(initial_music_idx(:).', 'stable');
    candidate_indices = candidate_indices(candidate_indices >= 1 & candidate_indices <= p.first_resolution);

    if p.source_count == 1
        if isempty(candidate_indices)
            [~, fallback_idx] = run_coarse_music(sensor_count, 1, x_music, p);
            candidate_indices = fallback_idx(1);
        end
        return;
    end

    if is_candidate_pool_confident(candidate_indices, sensor_count, p)
        candidate_indices = candidate_indices(1:p.source_count);
        return;
    end

    if p.enable_residual_augmentation
        coarse_pool_size = min(p.first_resolution, max(2 * p.source_count, p.source_count + 2));
        [~, music_idx] = run_coarse_music(sensor_count, coarse_pool_size, x_music, p);

        candidate_indices = unique([candidate_indices, music_idx(:).'], 'stable');
        candidate_indices = candidate_indices(candidate_indices >= 1 & candidate_indices <= p.first_resolution);

        if ~isempty(candidate_indices)
            residual_pool = [];
            primary_count = min(numel(candidate_indices), p.source_count);
            for ii = 1:primary_count
                coarse_phi = p.phi_list_first(candidate_indices(ii));
                steering = exp(1j * p.k * z_music * sin(coarse_phi));
                alpha = (steering' * x_music) / (steering' * steering);
                residual_x = x_music - steering * alpha;
                [~, residual_idx] = run_coarse_music(sensor_count, p.source_count, residual_x, p);
                residual_pool = [residual_pool, residual_idx(:).']; %#ok<AGROW>
            end
            candidate_indices = unique([candidate_indices, residual_pool], 'stable');
        end
    end

    candidate_indices = candidate_indices(candidate_indices >= 1 & candidate_indices <= p.first_resolution);

    if isempty(candidate_indices)
        [~, fallback_idx] = run_coarse_music(sensor_count, 1, x_music, p);
        candidate_indices = fallback_idx(1);
    end

    if numel(candidate_indices) < p.source_count
        [~, sorted_idx] = sort(compute_candidate_metric(1:p.first_resolution, sensor_count, p), 'descend');
        for ii = 1:numel(sorted_idx)
            candidate_idx = sorted_idx(ii);
            if any(candidate_indices == candidate_idx)
                continue;
            end
            if any(abs(candidate_indices - candidate_idx) <= 1)
                continue;
            end
            candidate_indices(end + 1) = candidate_idx; %#ok<AGROW>
            if numel(candidate_indices) >= p.source_count
                break;
            end
        end
    end

    candidate_indices = prune_candidate_pool(candidate_indices, sensor_count, p);
end

function [candidate_indices] = prune_candidate_pool(candidate_indices, sensor_count, p)
    candidate_indices = unique(candidate_indices(:).', 'stable');
    candidate_indices = candidate_indices(candidate_indices >= 1 & candidate_indices <= p.first_resolution);
    if ~p.enable_candidate_pruning
        return;
    end
    max_pool_size = min(p.first_resolution, max(2 * p.source_count, p.source_count + 2));
    if numel(candidate_indices) <= max_pool_size
        return;
    end

    candidate_scores = compute_candidate_metric(candidate_indices, sensor_count, p);
    [~, order] = sort(candidate_scores, 'descend');
    candidate_indices = candidate_indices(order(1:max_pool_size));
end

function [music_out, music_idx] = run_coarse_music(sensor_count, requested_count, x_music, p)
    [music_out, music_idx] = music_solver(1, p.first_resolution, requested_count, ...
        sensor_count, p.k, p.z1(1:sensor_count), x_music, 0);
end

function [is_confident] = is_candidate_pool_confident(candidate_indices, sensor_count, p)
    if numel(candidate_indices) < p.source_count
        is_confident = false;
        return;
    end

    z_music = p.z1(1:sensor_count);
    x_music = p.x1(1:sensor_count, :);
    is_confident = is_confident_coarse_set(candidate_indices(1:p.source_count), z_music, x_music, p);
end

function [is_confident] = is_confident_coarse_set(candidate_idx, z_music, x_music, p)
    candidate_idx = sort(candidate_idx(:));
    if numel(candidate_idx) < p.source_count
        is_confident = false;
        return;
    end

    min_separation_bins = max(3, ceil(p.first_resolution / 64));
    if any(diff(candidate_idx) < min_separation_bins)
        is_confident = false;
        return;
    end

    candidate_phi = p.phi_list_first(candidate_idx);
    residual = compute_joint_residual(candidate_phi, z_music, x_music, p);
    signal_energy = norm(x_music, 'fro')^2;
    if signal_energy <= 0
        is_confident = false;
        return;
    end

    residual_ratio = residual / signal_energy;
    is_confident = residual_ratio <= 0.35;
end

function [counts] = normalize_stage_counts(values, target_length)
    values = values(:).';
    if isempty(values)
        counts = ones(1, target_length);
        return;
    end
    counts = zeros(1, target_length);
    copy_len = min(numel(values), target_length);
    counts(1:copy_len) = values(1:copy_len);
    if copy_len < target_length
        counts(copy_len + 1:end) = values(copy_len);
    end
end

function [selected_idx] = select_ranked_peak(normalized_magnitude, peak_count, min_peak_height, anchor_idx)
    if max(normalized_magnitude) < min_peak_height
        [~, selected_idx] = max(normalized_magnitude);
        return;
    end

    peak_count = max(1, round(peak_count));
    [pks, ind] = findpeaks(normalized_magnitude, ...
        'SortStr', 'descend', 'NPeaks', peak_count);
    if isempty(ind)
        [~, selected_idx] = max(normalized_magnitude);
        return;
    end

    valid_idx = pks >= min_peak_height;
    if ~any(valid_idx)
        [~, selected_idx] = max(normalized_magnitude);
        return;
    end

    ind = ind(valid_idx);
    anchor_idx = min(max(round(anchor_idx), 1), numel(normalized_magnitude));
    [~, best_peak_idx] = min(abs(ind - anchor_idx));
    selected_idx = ind(best_peak_idx);
end

function [residual] = compute_residual(phi, p)
    sensor_count = min(p.stage_sensor_counts(min(p.step, numel(p.stage_sensor_counts))), p.sample_count);
    z_res = p.z1(1:sensor_count);
    y_res = p.x1(1:sensor_count, :);
    steering = exp(1j * p.k * z_res * sin(phi));
    alpha = (steering' * y_res) / (steering' * steering);
    residual = norm(y_res - steering * alpha, 'fro')^2;
end

function [refined_phi] = local_refine(phi_center, grid_spacing, p)
    if grid_spacing <= 0
        refined_phi = phi_center;
        return;
    end

    % Paper default: five 15-point zoom rounds. zoom_rounds is exposed
    % only so the experiment harness can reproduce the one-round
    % ablation without maintaining a second solver implementation.
    refined_phi = phi_center;
    search_radius = 1.5 * grid_spacing;
    n_samples = 15;
    for zoom_idx = 1:p.zoom_rounds
        phi_min = max(-pi / 2, refined_phi - search_radius);
        phi_max = min(pi / 2,  refined_phi + search_radius);
        phi_candidates = linspace(phi_min, phi_max, n_samples);
        if strcmp(p.local_refinement_mode, 'vectorized_grid')
            residuals = evaluate_single_residuals(phi_candidates, p);
        else
            residuals = zeros(size(phi_candidates));
            for kk = 1:numel(phi_candidates)
                residuals(kk) = compute_residual(phi_candidates(kk), p);
            end
        end
        [~, best_idx] = min(residuals);
        refined_phi = phi_candidates(best_idx);
        search_radius = search_radius * 2 / (n_samples - 1);
    end
end

function [residuals] = evaluate_single_residuals(phi_candidates, p)
    sensor_count = min( ...
        p.stage_sensor_counts(min(p.step, numel(p.stage_sensor_counts))), ...
        p.sample_count);
    z_eval = p.z1(1:sensor_count);
    y_eval = p.x1(1:sensor_count, :);
    steering = exp(1j * p.k * z_eval * sin(phi_candidates(:).'));
    projection = steering' * y_eval;
    steering_energy = sum(abs(steering) .^ 2, 1).';
    signal_energy = norm(y_eval, 'fro') ^ 2;
    residuals = signal_energy - ...
        sum(abs(projection) .^ 2, 2) ./ steering_energy;
    residuals = max(real(residuals), 0).';
end

function [accept_without_joint] = should_accept_without_joint(angle_deg_list, grid_spacing_list, p)
    if strcmp(p.joint_refinement_mode, 'never')
        accept_without_joint = true;
        return;
    end
    if strcmp(p.joint_refinement_mode, 'always')
        accept_without_joint = false;
        return;
    end
    accept_without_joint = p.fast_path_enabled && ...
        is_fast_selection_confident(angle_deg_list, grid_spacing_list, p);
end

function [is_confident] = is_fast_selection_confident(angle_deg_list, grid_spacing_list, p)
    angle_deg_list = sort(angle_deg_list(:));
    if numel(angle_deg_list) < p.source_count
        is_confident = false;
        return;
    end

    min_separation_deg = max(1.5 * max(grid_spacing_list) * 180 / pi, 3.0);
    if any(diff(angle_deg_list) < min_separation_deg)
        is_confident = false;
        return;
    end

    sensor_count = min(p.stage_sensor_counts(p.step), p.sample_count);
    z_eval = p.z1(1:sensor_count);
    x_eval = p.x1(1:sensor_count, :);
    signal_energy = norm(x_eval, 'fro')^2;
    if signal_energy <= 0
        is_confident = false;
        return;
    end

    residual = compute_joint_residual(angle_deg_list * pi / 180, z_eval, x_eval, p);
    residual_ratio = residual / signal_energy;
    is_confident = residual_ratio <= 0.08;
end

function [residual] = compute_joint_residual(phi_vec, z_joint, y_joint, p)
    steering = exp(1j * p.k * z_joint * sin(phi_vec.'));
    alpha = steering \ y_joint;
    residual = norm(y_joint - steering * alpha, 'fro')^2;
end

function [candidate_metric] = compute_candidate_metric(candidate_idx, sensor_count, p)
    candidate_phi = p.phi_list_first(candidate_idx);
    steering = exp(1j * p.k * p.z1(1:sensor_count) * sin(candidate_phi.'));
    projection = steering' * p.x1(1:sensor_count, :);
    candidate_metric = sqrt(sum(abs(projection) .^ 2, 2));
end

function out = shrink_complex(x, tau)
    out = sign(real(x)) .* max(abs(real(x)) - tau, 0) + ...
        1j * sign(imag(x)) .* max(abs(imag(x)) - tau, 0);
end

% Self-contained coarse MUSIC dependency.
function [out, idx] = music_solver(times, K, M, N, k, z, X1, display)

    if nargin < 8
        display = 0;
    end

    if isvector(X1)
        X1 = X1(:);
    end

    X1 = X1(1:N, :);
    z = z(1:N);
    num_snapshots = size(X1, 2);

    R = (X1 * X1') / num_snapshots;
    [EV, D] = eig(R);
    EVA = diag(D);
    [EVA, I] = sort(EVA, 'descend');
    Q = EV(:, I);
    Q_n = Q(:, M+1:N);

    phi_list = linspace(-pi/2, pi/2, K)';
    S1 = exp(1j * k * z * sin(phi_list'));
    P_MUSIC = 1 ./ sum(abs(Q_n' * S1).^2);

    P_MUSIC = abs(P_MUSIC);
    P_MUSIC_max = max(P_MUSIC);
    P_MUSIC_dB = 10 * log10(P_MUSIC / P_MUSIC_max);

    [P_peaks, P_peaks_idx] = findpeaks(P_MUSIC_dB);
    if ~isempty(P_peaks_idx)
        [P_peaks, I] = sort(P_peaks, 'descend');
        P_peaks_idx = P_peaks_idx(I);
    else
        P_peaks = [];
        P_peaks_idx = [];
    end

    if numel(P_peaks_idx) < M
        [sorted_spectrum, sorted_idx] = sort(P_MUSIC_dB, 'descend');
        for ii = 1:numel(sorted_idx)
            candidate_idx = sorted_idx(ii);
            if any(abs(P_peaks_idx - candidate_idx) <= 1)
                continue;
            end
            P_peaks_idx(end + 1, 1) = candidate_idx; %#ok<AGROW>
            P_peaks(end + 1, 1) = sorted_spectrum(ii); %#ok<AGROW>
            if numel(P_peaks_idx) >= M
                break;
            end
        end
    end

    peak_count = min(M, numel(P_peaks_idx));
    P_peaks = P_peaks(1:peak_count);
    P_peaks_idx = P_peaks_idx(1:peak_count);
    phi_e = phi_list(P_peaks_idx) * 180 / pi;

    if display == 1
        figure;
        plot(P_MUSIC_dB);
    end

    out = phi_e;
    idx = P_peaks_idx;

end
