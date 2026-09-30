function [out, index, diagnostics] = hadmm_solver_diagnostic(kmax, k, z, K, factor, M, N, varargin)
%HADMM_SOLVER MUSIC-guided hierarchical local solver.
% Optional experiment controls live in the final opts struct. Their defaults
% reproduce the paper_zoom5_clean_v1 path; diagnostics are enabled by default.

    fast_path_enabled = true;
    zoom_rounds = 5;
    local_solver = 'admm_s';
    admm_iterations = kmax;
    enable_subarray_expansion = true;
    enable_residual_augmentation = true;
    enable_candidate_pruning = true;
    joint_refinement_mode = 'gated';
    joint_mode_explicit = false;
    collect_diagnostics = true;
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
        if isfield(opts, 'collect_diagnostics')
            collect_diagnostics = logical(opts.collect_diagnostics);
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

    d.timing_coarse_music = 0;
    d.timing_candidate_enhancement = 0;
    d.timing_local_matrix = zeros(1, step);
    d.timing_gram_projection = zeros(1, step);
    d.timing_factorization = zeros(1, step);
    d.timing_solver_updates = zeros(1, step);
    d.timing_zoom = 0;
    d.timing_final_selection = 0;
    d.timing_joint_refinement = 0;
    d.zoom_evaluation_count = 0;
    d.music_call_count = 0;
    d.candidate_pool_history = struct('sensor_count', {}, 'candidate_indices', {});
    d.initial_candidate_indices = [];
    d.subarray_expansion_triggered = false;
    d.residual_augmentation_triggered = false;
    d.candidate_pruning_triggered = false;
    d.fast_acceptance_triggered = false;
    d.joint_refinement_triggered = false;
    d.candidate_subset_count = 0;
    d.selected_prezoom_angles_deg = [];
    if collect_diagnostics
        total_timer = tic;
    else
        total_timer = [];
    end

    first_resolution = factor(1);
    phi_list_first = linspace(-pi/2, pi/2, first_resolution)';
    music_sensor_count = min([max(1, music_N), sample_count, numel(z1)]);

    % Explicit inputs for the ordinary local helpers below.
    p = struct;
    p.collect_diagnostics = collect_diagnostics;
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
    candidate_pool_timer = start_timer(p);
    [candidate_indices, d] = build_candidate_pool(p, d);
    candidate_pool_elapsed = stop_timer(candidate_pool_timer, p);
    if collect_diagnostics && (d.subarray_expansion_triggered || ...
            d.residual_augmentation_triggered || d.candidate_pruning_triggered || ...
            d.music_call_count > 1)
        d.timing_candidate_enhancement = max( ...
            candidate_pool_elapsed - d.timing_coarse_music, 0);
    end
    candidate_count = numel(candidate_indices);

    candidate_paths = zeros(candidate_count, step);
    candidate_angles = zeros(candidate_count, 1);
    candidate_residuals = inf(candidate_count, 1);
    candidate_grid_spacings = zeros(candidate_count, 1);
    d.candidate_prezoom_angles = zeros(candidate_count, 1);
    d.candidate_window_widths = nan(candidate_count, step);
    d.candidate_stage_diagnostics = cell(candidate_count, step);

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
            matrix_timer = start_timer(p);
            S1_mini = exp(1j * k * z_stage * sin(phi_last(local_indices).'));
            d.timing_local_matrix(stage) = d.timing_local_matrix(stage) + ...
                stop_timer(matrix_timer, p);
            anchor_global_idx = factor(stage) * window_parent_idx;
            anchor_local_idx = anchor_global_idx - col_start + 1;

            K_local = size(S1_mini, 2);
            snapshot_count = size(y_stage, 2);
            primal_residual = nan(1, admm_iterations);
            dual_residual = nan(1, admm_iterations);

            if strcmp(local_solver, 'matched_filter')
                projection_timer = start_timer(p);
                ranking_state = S1_mini' * y_stage;
                d.timing_gram_projection(stage) = d.timing_gram_projection(stage) + ...
                    stop_timer(projection_timer, p);
                ranking_variable = 'matched_filter_projection';
            else
                gram_timer = start_timer(p);
                system_matrix = 2 .* (S1_mini' * S1_mini) + ...
                    rho .* eye(K_local);
                a = 2 .* S1_mini' * y_stage;
                d.timing_gram_projection(stage) = d.timing_gram_projection(stage) + ...
                    stop_timer(gram_timer, p);

                factor_timer = start_timer(p);
                B = inv(system_matrix);
                d.timing_factorization(stage) = d.timing_factorization(stage) + ...
                    stop_timer(factor_timer, p);

                if strcmp(local_solver, 'ridge')
                    update_timer = start_timer(p);
                    ranking_state = B * a;
                    d.timing_solver_updates(stage) = d.timing_solver_updates(stage) + ...
                        stop_timer(update_timer, p);
                    ranking_variable = 'ridge_s';
                else
                    sk = zeros(K_local, snapshot_count);
                    zk = zeros(K_local, snapshot_count);
                    uk = zeros(K_local, snapshot_count);
                    if collect_diagnostics
                        update_elapsed = 0;
                        for iter = 1:admm_iterations
                            update_timer = tic;
                            b = a + rho .* (zk - uk);
                            sk1 = B * b;

                            temp = sk1 + uk;
                            zk1 = shrink_complex(temp, tau);
                            uk1 = uk + sk1 - zk1;
                            update_elapsed = update_elapsed + toc(update_timer);
                            primal_residual(iter) = norm(sk1 - zk1, 'fro');
                            dual_residual(iter) = rho * norm(zk1 - zk, 'fro');
                            sk = sk1;
                            zk = zk1;
                            uk = uk1;
                        end
                    else
                        update_elapsed = 0;
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
                    end
                    d.timing_solver_updates(stage) = d.timing_solver_updates(stage) + ...
                        update_elapsed;
                    if strcmp(local_solver, 'admm_z')
                        ranking_state = zk;
                        ranking_variable = 'admm_z';
                    else
                        ranking_state = sk;
                        ranking_variable = 'admm_s';
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

            if collect_diagnostics
                stage_diagnostic = struct( ...
                    'local_solver', local_solver, ...
                    'iterations', admm_iterations, ...
                    'ranking_variable', ranking_variable, ...
                    'ranking_magnitude', magnitude, ...
                    'normalized_ranking', sk_stem, ...
                    'selected_local_index', selected_idx, ...
                    'anchor_local_index', anchor_local_idx, ...
                    'primal_residual', primal_residual, ...
                    'dual_residual', dual_residual);
            else
                stage_diagnostic = [];
            end
            local_idx = selected_idx;

            path_idx(stage) = local_indices(local_idx);
            if collect_diagnostics
                d.candidate_window_widths(cand, stage) = numel(local_indices);
                stage_diagnostic.sensor_count = sensor_count;
                stage_diagnostic.window_width = numel(local_indices);
                stage_diagnostic.local_global_indices = local_indices;
                stage_diagnostic.selected_global_index = path_idx(stage);
                d.candidate_stage_diagnostics{cand, stage} = stage_diagnostic;
            end
        end

        final_phi = phi_last(path_idx(step));
        if numel(phi_last) > 1
            grid_spacing = phi_last(2) - phi_last(1);
        else
            grid_spacing = 0;
        end
        final_grid_spacing = grid_spacing;
        prezoom_angle_deg = final_phi * 180 / pi;

        [refined_phi, d] = local_refine(final_phi, grid_spacing, p, d);
        angle_deg = refined_phi * 180 / pi;
        residual = compute_residual(refined_phi, p);
        candidate_paths(cand, :) = path_idx;
        candidate_angles(cand) = angle_deg;
        candidate_residuals(cand) = residual;
        candidate_grid_spacings(cand) = final_grid_spacing;
        d.candidate_prezoom_angles(cand) = prezoom_angle_deg;

    end

    %% Final source-set selection
    if source_count == 1
        [~, best_idx] = min(candidate_residuals);
        out = candidate_angles(best_idx);
        index = candidate_paths(best_idx, :);
        d.candidate_subset_count = candidate_count;
        d.selected_prezoom_angles_deg = d.candidate_prezoom_angles(best_idx);
    else
        all_angles = candidate_angles;
        all_paths = candidate_paths;
        all_grid_spacings = candidate_grid_spacings;
        % One-pass selection block: an early break accepts the current source set.
        while true
            selection_timer = start_timer(p);
            all_angles = all_angles(:);
            subset_count = min(source_count, numel(all_angles));

            if numel(all_angles) == subset_count
                selected_candidate_idx = (1:numel(all_angles)).';
                selected_angles = all_angles(:).';
                selected_paths = all_paths;
                [selected_angles, sort_idx] = sort(selected_angles, 'ascend');
                selected_paths = selected_paths(sort_idx, :);
                selected_grid_spacings = all_grid_spacings(sort_idx);
                selected_candidate_idx = selected_candidate_idx(sort_idx);
                d.candidate_subset_count = 1;
                d.selected_prezoom_angles_deg = ...
                    d.candidate_prezoom_angles(selected_candidate_idx).';
                [accept_without_joint, d] = should_accept_without_joint(selected_angles, selected_grid_spacings, p, d);

                if accept_without_joint
                    d.timing_final_selection = d.timing_final_selection + ...
                        stop_timer(selection_timer, p);
                    break;
                end
            end

            combinations = nchoosek(1:numel(all_angles), subset_count);
            d.candidate_subset_count = size(combinations, 1);
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
            selected_candidate_idx = best_subset(:);
            [selected_angles, sort_idx] = sort(selected_angles, 'ascend');
            selected_paths = selected_paths(sort_idx, :);
            selected_grid_spacings = selected_grid_spacings(sort_idx);
            selected_candidate_idx = selected_candidate_idx(sort_idx);
            d.selected_prezoom_angles_deg = ...
                d.candidate_prezoom_angles(selected_candidate_idx).';

            [accept_without_joint, d] = should_accept_without_joint(selected_angles, selected_grid_spacings, p, d);

            if accept_without_joint
                selected_angles = selected_angles(:).';
                d.timing_final_selection = d.timing_final_selection + ...
                    stop_timer(selection_timer, p);
                break;
            end

            d.timing_final_selection = d.timing_final_selection + stop_timer(selection_timer, p);
            d.joint_refinement_triggered = true;
            joint_timer = start_timer(p);
            zoom_time_before = d.timing_zoom;
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
                [best_pair(ii), d] = local_refine(best_pair(ii), joint_spacing, p, d);
            end
            refined_angles = best_pair * 180 / pi;

            for ii = 1:subset_count
                refined_residuals(ii) = compute_residual(best_pair(ii), p);
                [~, nearest_idx] = min(abs(final_phi_list - best_pair(ii)));
                refined_paths(ii, step) = nearest_idx;
            end
            joint_elapsed = stop_timer(joint_timer, p);
            d.timing_joint_refinement = d.timing_joint_refinement + ...
                max(joint_elapsed - (d.timing_zoom - zoom_time_before), 0);
            selected_angles = refined_angles;
            selected_paths = refined_paths;

            selected_angles = selected_angles(:).';
            break;
        end
        out = selected_angles;
        index = selected_paths;

    end

    normalized_options = struct( ...
        'local_solver', local_solver, ...
        'admm_iterations', admm_iterations, ...
        'enable_subarray_expansion', enable_subarray_expansion, ...
        'enable_residual_augmentation', enable_residual_augmentation, ...
        'enable_candidate_pruning', enable_candidate_pruning, ...
        'joint_refinement_mode', joint_refinement_mode, ...
        'forced_local_windows', ~isempty(forced_branch_paths), ...
        'collect_diagnostics', collect_diagnostics, ...
        'local_refinement_mode', local_refinement_mode, ...
        'zoom_rounds', zoom_rounds);

    if collect_diagnostics
        total_elapsed = toc(total_timer);
        component_sum = d.timing_coarse_music + d.timing_candidate_enhancement + ...
            sum(d.timing_local_matrix) + sum(d.timing_gram_projection) + ...
            sum(d.timing_factorization) + sum(d.timing_solver_updates) + ...
            d.timing_zoom + d.timing_final_selection + d.timing_joint_refinement;
        diagnostics = struct( ...
            'enabled', true, ...
            'options', normalized_options, ...
            'stage1_initial_candidates', d.initial_candidate_indices, ...
            'stage1_final_candidates', candidate_indices, ...
            'stage1_candidate_history', d.candidate_pool_history, ...
            'candidate_pool_size', candidate_count, ...
            'candidate_subset_count', d.candidate_subset_count, ...
            'branch_paths', candidate_paths, ...
            'branch_window_widths', d.candidate_window_widths, ...
            'branch_prezoom_angles_deg', d.candidate_prezoom_angles, ...
            'branch_postzoom_angles_deg', candidate_angles, ...
            'branch_stage_diagnostics', {d.candidate_stage_diagnostics}, ...
            'selected_prezoom_angles_deg', d.selected_prezoom_angles_deg, ...
            'selected_angles_deg', out, ...
            'selected_paths', index, ...
            'triggers', struct( ...
                'subarray_expansion', d.subarray_expansion_triggered, ...
                'residual_augmentation', d.residual_augmentation_triggered, ...
                'candidate_pruning', d.candidate_pruning_triggered, ...
                'fast_acceptance', d.fast_acceptance_triggered, ...
                'joint_refinement', d.joint_refinement_triggered), ...
            'timing_seconds', struct( ...
                'coarse_music', d.timing_coarse_music, ...
                'candidate_enhancement', d.timing_candidate_enhancement, ...
                'local_matrix_by_stage', d.timing_local_matrix, ...
                'gram_projection_by_stage', d.timing_gram_projection, ...
                'factorization_by_stage', d.timing_factorization, ...
                'solver_updates_by_stage', d.timing_solver_updates, ...
                'zoom', d.timing_zoom, ...
                'final_selection', d.timing_final_selection, ...
                'joint_refinement', d.timing_joint_refinement, ...
                'component_sum', component_sum, ...
                'other', max(total_elapsed - component_sum, 0), ...
                'total', total_elapsed), ...
            'stage_sensor_counts', stage_sensor_counts, ...
            'stage_peak_counts', stage_peak_counts, ...
            'zoom_evaluation_count', d.zoom_evaluation_count);
    else
        diagnostics = struct('enabled', false, 'options', normalized_options);
    end

end

% Ordinary local functions: explicit input parameters; d returns recording state.
function [candidate_indices, d] = build_candidate_pool(p, d)
    [base_pool, d] = build_candidate_pool_for_sensor_count(p.music_sensor_count, p, d);
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
        d.subarray_expansion_triggered = true;
        expanded_music_sensor_count = sensor_schedule(schedule_idx);
        [expanded_pool, d] = build_candidate_pool_for_sensor_count(expanded_music_sensor_count, p, d);
        if is_candidate_pool_confident(expanded_pool, expanded_music_sensor_count, p)
            candidate_indices = expanded_pool;
            return;
        end
        [candidate_indices, d] = prune_candidate_pool([expanded_pool, candidate_indices], expanded_music_sensor_count, p, d);
    end
end

function [candidate_indices, d] = build_candidate_pool_for_sensor_count(sensor_count, p, d)
    z_music = p.z1(1:sensor_count);
    x_music = p.x1(1:sensor_count, :);

    [~, initial_music_idx, d] = run_coarse_music(sensor_count, p.source_count, x_music, p, d);

    candidate_indices = unique(initial_music_idx(:).', 'stable');
    candidate_indices = candidate_indices(candidate_indices >= 1 & candidate_indices <= p.first_resolution);
    if isempty(d.initial_candidate_indices)
        d.initial_candidate_indices = candidate_indices;
    end

    if p.source_count == 1
        if isempty(candidate_indices)
            [~, fallback_idx, d] = run_coarse_music(sensor_count, 1, x_music, p, d);
            candidate_indices = fallback_idx(1);
        end
        d = record_candidate_pool(sensor_count, candidate_indices, p, d);
        return;
    end

    if is_candidate_pool_confident(candidate_indices, sensor_count, p)
        candidate_indices = candidate_indices(1:p.source_count);
        d = record_candidate_pool(sensor_count, candidate_indices, p, d);
        return;
    end

    if p.enable_residual_augmentation
        d.residual_augmentation_triggered = true;
        coarse_pool_size = min(p.first_resolution, max(2 * p.source_count, p.source_count + 2));
        [~, music_idx, d] = run_coarse_music(sensor_count, coarse_pool_size, x_music, p, d);

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
                [~, residual_idx, d] = run_coarse_music(sensor_count, p.source_count, residual_x, p, d);
                residual_pool = [residual_pool, residual_idx(:).']; %#ok<AGROW>
            end
            candidate_indices = unique([candidate_indices, residual_pool], 'stable');
        end
    end

    candidate_indices = candidate_indices(candidate_indices >= 1 & candidate_indices <= p.first_resolution);

    if isempty(candidate_indices)
        [~, fallback_idx, d] = run_coarse_music(sensor_count, 1, x_music, p, d);
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

    [candidate_indices, d] = prune_candidate_pool(candidate_indices, sensor_count, p, d);
    d = record_candidate_pool(sensor_count, candidate_indices, p, d);
end

function [candidate_indices, d] = prune_candidate_pool(candidate_indices, sensor_count, p, d)
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
    d.candidate_pruning_triggered = true;
end

function [music_out, music_idx, d] = run_coarse_music(sensor_count, requested_count, x_music, p, d)
    music_timer = start_timer(p);
    [music_out, music_idx] = music_solver(1, p.first_resolution, requested_count, ...
        sensor_count, p.k, p.z1(1:sensor_count), x_music, 0);
    music_elapsed = stop_timer(music_timer, p);
    d.music_call_count = d.music_call_count + 1;
    if d.music_call_count == 1
        d.timing_coarse_music = d.timing_coarse_music + music_elapsed;
    end
end

function [d] = record_candidate_pool(sensor_count, candidate_indices, p, d)
    if ~p.collect_diagnostics
        return;
    end
    d.candidate_pool_history(end + 1) = struct( ...
        'sensor_count', sensor_count, ...
        'candidate_indices', candidate_indices(:).'); %#ok<AGROW>
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

function [refined_phi, d] = local_refine(phi_center, grid_spacing, p, d)
    zoom_timer = start_timer(p);
    if grid_spacing <= 0
        refined_phi = phi_center;
        d.timing_zoom = d.timing_zoom + stop_timer(zoom_timer, p);
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
        d.zoom_evaluation_count = d.zoom_evaluation_count + numel(phi_candidates);
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
    d.timing_zoom = d.timing_zoom + stop_timer(zoom_timer, p);
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

function [accept_without_joint, d] = should_accept_without_joint(angle_deg_list, grid_spacing_list, p, d)
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
    if accept_without_joint
        d.fast_acceptance_triggered = true;
    end
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

function [timer_handle] = start_timer(p)
    if p.collect_diagnostics
        timer_handle = tic;
    else
        timer_handle = [];
    end
end

function [elapsed] = stop_timer(timer_handle, p)
    if p.collect_diagnostics
        elapsed = toc(timer_handle);
    else
        elapsed = 0;
    end
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
