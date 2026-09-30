function [out, index] = hadmm_solver_paper(Y, source_count)
%HADMM_SOLVER_PAPER Fixed paper configuration; Y is sensors-by-snapshots.
% Read the main function in order. All helpers follow its closing end.
    if size(Y,1)<64 || ~ismember(source_count,[1,2])
        error('Use at least 64 sensors and one or two sources.');
    end
    k=2*pi/0.03; sensor_count=size(Y,1);
    z=(0:sensor_count-1).' .* 0.015;
    factor=[256,2,2,2]; stages=length(factor); music_N=64;
    neighbor=4; threshold=0.60; lambda=0.30; rho=1;
    tau=lambda/rho; admm_iterations=2; zoom_rounds=5;
    coarse_grid=linspace(-pi/2,pi/2,factor(1))';

    %% 1. MUSIC candidates, with evidence-based subarray expansion
    candidates=coarse_candidates(Y(1:music_N,:),z(1:music_N),k,source_count,coarse_grid);
    if source_count>1 && ~coarse_confident(candidates,Y(1:music_N,:), ...
            z(1:music_N),k,source_count,coarse_grid)
        sensor_schedule=unique([music_N,min(sensor_count,music_N+8),min(sensor_count,music_N+16)]);
        sensor_schedule=sensor_schedule(sensor_schedule>music_N);
        for expanded_N=sensor_schedule
            coarse_Y=Y(1:expanded_N,:); coarse_z=z(1:expanded_N);
            expanded=coarse_candidates(coarse_Y,coarse_z,k,source_count,coarse_grid);
            if coarse_confident(expanded,coarse_Y,coarse_z,k,source_count,coarse_grid)
                candidates=expanded;
                break;
            end
            candidates=prune_candidates([expanded,candidates],coarse_Y,coarse_z,k,source_count,coarse_grid);
        end
    end

    %% 2. Hierarchical local dictionaries and two ADMM updates per level
    count=numel(candidates);
    paths=zeros(count,stages); angles=zeros(count,1);
    residuals=inf(count,1); spacings=zeros(count,1);
    for candidate=1:count
        path=zeros(1,stages); path(1)=candidates(candidate);
        resolution=factor(1);
        for stage=2:stages
            resolution=resolution*factor(stage);
            parent=path(stage-1);
            upper=min(parent+neighbor,resolution/factor(stage));
            lower=max(parent-neighbor,0);
            grid=linspace(-pi/2,pi/2,resolution)';
            first=factor(stage)*lower+1; last=factor(stage)*upper;
            local_indices=first:last;
            A=exp(1j*k*z*sin(grid(local_indices).'));
            anchor=factor(stage)*parent-first+1;

            columns=size(A,2); snapshots=size(Y,2);
            system_matrix=2.*(A'*A)+rho.*eye(columns);
            a=2.*A'*Y;
            B=inv(system_matrix);
            s=zeros(columns,snapshots); v=zeros(columns,snapshots); u=zeros(columns,snapshots);
            for iteration=1:admm_iterations
                b=a+rho.*(v-u);
                s_next=B*b;
                temp=s_next+u;
                v_next=sign(real(temp)).*max(abs(real(temp))-tau,0) ...
                    +1j*sign(imag(temp)).*max(abs(imag(temp))-tau,0);
                u_next=u+s_next-v_next;
                s=s_next; v=v_next; u=u_next;
            end

            % Rank the dense ADMM variable s, not the thresholded variable v.
            magnitude=sqrt(sum(abs(s).^2,2));
            if max(magnitude)>0, magnitude=magnitude/max(magnitude); end
            if max(magnitude)<threshold
                [~,selected]=max(magnitude);
            else
                [peaks,locations]=findpeaks(magnitude,'SortStr','descend','NPeaks',1);
                valid=peaks>=threshold;
                if isempty(locations) || ~any(valid)
                    [~,selected]=max(magnitude);
                else
                    locations=locations(valid);
                    anchor=min(max(round(anchor),1),numel(magnitude));
                    [~,nearest]=min(abs(locations-anchor));
                    selected=locations(nearest);
                end
            end
            path(stage)=local_indices(selected);
        end

        %% 3. Five-round residual zoom for this candidate
        spacing=grid(2)-grid(1);
        phi=residual_zoom(grid(path(stages)),spacing,Y,z,k,zoom_rounds);
        paths(candidate,:)=path;
        angles(candidate)=phi*180/pi;
        spacings(candidate)=spacing;
        steering=exp(1j*k*z*sin(phi));
        amplitude=(steering'*Y)/(steering'*steering);
        residuals(candidate)=norm(Y-steering*amplitude,'fro')^2;
    end

    %% 4. Select the source set; accept directly when the residual is convincing
    if source_count==1
        [~,best]=min(residuals);
        out=angles(best); index=paths(best,:);
        return;
    end
    subset_count=min(source_count,numel(angles));
    if numel(angles)==subset_count
        [out,order]=sort(angles(:).','ascend');
        index=paths(order,:); selected_spacings=spacings(order);
        if final_confident(out,selected_spacings,Y,z,k,source_count), return; end
    end

    combinations=nchoosek(1:numel(angles),subset_count);
    minimum_separation=1.5*max(spacings)*180/pi;
    best_residual=inf; best_subset=combinations(1,:);
    for combination=1:size(combinations,1)
        subset=combinations(combination,:);
        current_angles=sort(angles(subset));
        if subset_count>1 && any(diff(current_angles)<minimum_separation), continue; end
        value=joint_residual(current_angles*pi/180,Y,z,k);
        if value<best_residual
            best_residual=value; best_subset=subset;
        end
    end
    out=angles(best_subset); index=paths(best_subset,:);
    selected_spacings=spacings(best_subset);
    [out,order]=sort(out,'ascend');
    index=index(order,:); selected_spacings=selected_spacings(order);
    if final_confident(out,selected_spacings,Y,z,k,source_count)
        out=out(:).';
        return;
    end

    %% 5. If needed, jointly search the pair, then apply the same residual zoom
    pair=out(:)*pi/180;
    spacing=max(selected_spacings);
    if spacing<=0, spacing=pi/max(prod(factor),1); end
    radius=3*spacing; search_points=cell(subset_count,1);
    for source=1:subset_count
        search_points{source}=linspace(max(-pi/2,pair(source)-radius), ...
            min(pi/2,pair(source)+radius),11);
    end
    best_residual=inf; best_pair=pair;
    if subset_count==2
        for left=search_points{1}
            for right=search_points{2}
                if right<=left+spacing, continue; end
                value=joint_residual([left;right],Y,z,k);
                if value<best_residual
                    best_residual=value; best_pair=[left;right];
                end
            end
        end
    end
    final_grid=linspace(-pi/2,pi/2,prod(factor))';
    for source=1:subset_count
        best_pair(source)=residual_zoom(best_pair(source),spacing,Y,z,k,zoom_rounds);
    end
    out=(best_pair*180/pi).';
    for source=1:subset_count
        [~,nearest]=min(abs(final_grid-best_pair(source)));
        index(source,stages)=nearest;
    end
end

% Local helpers below are independent: all required data are explicit inputs.
function candidates=coarse_candidates(Y,z,k,M,grid)
    resolution=numel(grid); N=size(Y,1);
    [~,initial]=music_solver(resolution,M,N,k,z,Y);
    candidates=unique(initial(:).','stable');
    candidates=candidates(candidates>=1 & candidates<=resolution);
    if M==1
        if isempty(candidates)
            [~,fallback]=music_solver(resolution,1,N,k,z,Y); candidates=fallback(1);
        end
        return;
    end
    if coarse_confident(candidates,Y,z,k,M,grid)
        candidates=candidates(1:M);
        return;
    end

    % Enlarge an uncertain pool using MUSIC on the observation and its residual.
    pool_size=min(resolution,max(2*M,M+2));
    [~,extra]=music_solver(resolution,pool_size,N,k,z,Y);
    candidates=unique([candidates,extra(:).'],'stable');
    candidates=candidates(candidates>=1 & candidates<=resolution);
    if ~isempty(candidates)
        residual_pool=[];
        for source=1:min(numel(candidates),M)
            steering=exp(1j*k*z*sin(grid(candidates(source))));
            amplitude=(steering'*Y)/(steering'*steering);
            residual_Y=Y-steering*amplitude;
            [~,extra]=music_solver(resolution,M,N,k,z,residual_Y);
            residual_pool=[residual_pool,extra(:).']; %#ok<AGROW>
        end
        candidates=unique([candidates,residual_pool],'stable');
    end
    candidates=candidates(candidates>=1 & candidates<=resolution);
    if isempty(candidates)
        [~,fallback]=music_solver(resolution,1,N,k,z,Y); candidates=fallback(1);
    end
    if numel(candidates)<M
        [~,order]=sort(candidate_metric(1:resolution,Y,z,k,grid),'descend');
        for i=1:numel(order)
            candidate=order(i);
            if any(candidates==candidate) || any(abs(candidates-candidate)<=1), continue; end
            candidates(end+1)=candidate; %#ok<AGROW>
            if numel(candidates)>=M, break; end
        end
    end
    candidates=prune_candidates(candidates,Y,z,k,M,grid);
end

function candidates=prune_candidates(candidates,Y,z,k,M,grid)
    candidates=unique(candidates(:).','stable');
    candidates=candidates(candidates>=1 & candidates<=numel(grid));
    limit=min(numel(grid),max(2*M,M+2));
    if numel(candidates)<=limit, return; end
    scores=candidate_metric(candidates,Y,z,k,grid);
    [~,order]=sort(scores,'descend');
    candidates=candidates(order(1:limit));
end

function confident=coarse_confident(candidates,Y,z,k,M,grid)
    confident=false;
    if numel(candidates)<M, return; end
    selected=sort(candidates(1:M));
    selected=selected(:);
    if any(diff(selected)<max(3,ceil(numel(grid)/64))), return; end
    value=joint_residual(grid(selected),Y,z,k);
    energy=norm(Y,'fro')^2;
    if energy>0, confident=value/energy<=0.35; end
end

function phi=residual_zoom(phi,spacing,Y,z,k,rounds)
    if spacing<=0, return; end
    radius=1.5*spacing; samples=15;
    for iteration=1:rounds
        points=linspace(max(-pi/2,phi-radius),min(pi/2,phi+radius),samples);
        steering=exp(1j*k*z*sin(points(:).'));
        projection=steering'*Y;
        steering_energy=sum(abs(steering).^2,1).';
        signal_energy=norm(Y,'fro')^2;
        residuals=signal_energy-sum(abs(projection).^2,2)./steering_energy;
        residuals=max(real(residuals),0).';
        [~,best]=min(residuals);
        phi=points(best);
        radius=radius*2/(samples-1);
    end
end

function confident=final_confident(angles,spacings,Y,z,k,M)
    angles=sort(angles(:)); confident=false;
    if numel(angles)<M, return; end
    separation=max(1.5*max(spacings)*180/pi,3.0);
    if any(diff(angles)<separation), return; end
    energy=norm(Y,'fro')^2;
    if energy<=0, return; end
    value=joint_residual(angles*pi/180,Y,z,k);
    confident=value/energy<=0.08;
end

function value=joint_residual(phi,Y,z,k)
    steering=exp(1j*k*z*sin(phi.'));
    amplitude=steering\Y;
    value=norm(Y-steering*amplitude,'fro')^2;
end

function scores=candidate_metric(indices,Y,z,k,grid)
    phi=grid(indices);
    steering=exp(1j*k*z*sin(phi.'));
    projection=steering'*Y;
    scores=sqrt(sum(abs(projection).^2,2));
end

function [angles,indices]=music_solver(resolution,M,N,k,z,Y)
    if isvector(Y), Y=Y(:); end
    Y=Y(1:N,:); z=z(1:N);
    covariance=(Y*Y')/size(Y,2);
    [vectors,values]=eig(covariance);
    [~,order]=sort(diag(values),'descend');
    vectors=vectors(:,order);
    noise_space=vectors(:,M+1:N);
    grid=linspace(-pi/2,pi/2,resolution)';
    steering=exp(1j*k*z*sin(grid'));
    spectrum=1./sum(abs(noise_space'*steering).^2);
    spectrum=abs(spectrum);
    spectrum_db=10*log10(spectrum/max(spectrum));
    [peaks,indices]=findpeaks(spectrum_db);
    if ~isempty(indices)
        [peaks,order]=sort(peaks,'descend');
        indices=indices(order);
    else
        peaks=[]; indices=[];
    end
    if numel(indices)<M
        [sorted_spectrum,order]=sort(spectrum_db,'descend');
        for i=1:numel(order)
            candidate=order(i);
            if any(abs(indices-candidate)<=1), continue; end
            indices(end+1,1)=candidate; %#ok<AGROW>
            peaks(end+1,1)=sorted_spectrum(i); %#ok<AGROW>
            if numel(indices)>=M, break; end
        end
    end
    indices=indices(1:min(M,numel(indices)));
    angles=grid(indices)*180/pi;
end
