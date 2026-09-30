function summary = test_hadmm(mode, output_dir, N_list, snr_list)
% Run a quick test, or test_hadmm('full') for the paper experiment.
    if nargin<1, mode='smoke'; end
    solver=@hadmm_solver;
    if strcmp(mode,'paper') || strcmp(mode,'paper-full')
        solver=@hadmm_solver_paper;
        if strcmp(mode,'paper'), mode='smoke'; else, mode='full'; end
    end
    all_N=512:256:2048; all_snr=-25:5:40; seed_slots=[1,1,2,3,4,5,6];
    base_seed=21910000; trials=200;
    if nargin<3, N_list=all_N; end
    if nargin<4, snr_list=all_snr; end
    if strcmp(mode,'smoke')
        trials=2;
        if nargin<3, N_list=512; end
        if nargin<4, snr_list=20; end
    elseif ~strcmp(mode,'full')
        error('Use smoke, full, paper, or paper-full.');
    end
    d=0.015; k=2*pi/0.03; factor=[256,2,2,2]; music_N=64;
    neighbor=4; threshold=0.60; lambda=0.30; rho=1;
    opts=struct('zoom_rounds',5,'local_solver','admm_s', ...
        'admm_iterations',2,'local_refinement_mode','vectorized_grid');
    [valid_N,N_indices]=ismember(N_list,all_N);
    [valid_snr,snr_indices]=ismember(snr_list,all_snr);
    if ~all(valid_N) || ~all(valid_snr), error('Use subsets of all_N and all_snr.'); end
    if nargin<2, output_dir=['results_' datestr(now,'yyyymmdd_HHMMSS')]; end
    if exist(output_dir,'file'), error('Use a new output directory.'); end
    findpeaks([0,1,0]); % Check the toolbox before starting the experiment.
    mkdir(output_dir);
    save(fullfile(output_dir,'config.mat')); % Parameters only; no source/environment archive.
    rows=[];

    % Generate an observation, solve it, and record error and elapsed time.
    for M=1:2
        for ni=1:numel(N_list)
            N=N_list(ni); z=(0:N-1).' .* d;
            for si=1:numel(snr_list)
                snr_db=snr_list(si);
                seeds=base_seed+M*1000000+seed_slots(N_indices(ni))*100000 ...
                    +snr_indices(si)*1000+(1:trials);
                truth=zeros(trials,M); estimates=nan(trials,M); errors=180*ones(trials,M);
                runtime=zeros(trials,1); invalid=false(trials,1);
                for trial=1:trials
                    [Y,truth(trial,:)]=observation(seeds(trial),M,z,k,snr_db);
                    args={2,k,z,size(Y,2),factor,ones(1,6),N*ones(1,6), ...
                        music_N,neighbor*ones(1,6),threshold*ones(1,6),lambda,rho,Y,M,opts};
                    if isequal(solver,@hadmm_solver_paper), args={Y,M}; end
                    if trial==1, solver(args{:}); end % Untimed warmup.
                    started=tic;
                    try
                        angle=solver(args{:});
                        runtime(trial)=toc(started);
                        invalid(trial)=numel(angle)~=M || any(~isfinite(angle));
                        if ~invalid(trial)
                            estimates(trial,:)=sort(angle(:).');
                            errors(trial,:)=estimates(trial,:)-sort(truth(trial,:));
                        end
                    catch failure
                        runtime(trial)=toc(started);
                        if contains(lower(failure.identifier),'license'), rethrow(failure); end
                        invalid(trial)=true; % Keep the 180-degree penalty; never drop a trial.
                    end
                end
                rmse=sqrt(mean(errors.^2,'all')); milliseconds=1000*mean(runtime);
                rows(end+1,:)=[M,N,snr_db,trials,rmse,milliseconds,sum(invalid)]; %#ok<AGROW>
                save(fullfile(output_dir,sprintf('M%d_N%d_snr%+d.mat',M,N,snr_db)), ...
                    'M','N','snr_db','seeds','truth','estimates','errors','runtime','invalid');
                fprintf('M=%d N=%d SNR=%+d: RMSE=%.6g deg, runtime=%.4f ms\n', ...
                    M,N,snr_db,rmse,milliseconds);
            end
        end
    end
    summary=array2table(rows,'VariableNames',{'sources','N','snr_db','trials', ...
        'rmse_deg','mean_runtime_ms','invalid_count'});
    writetable(summary,fullfile(output_dir,'summary.csv'));

    % Show and save log10-RMSE and runtime versus SNR for both source counts.
    fig=figure('Color','w','Position',[100,100,1100,700]); tiledlayout(2,2);
    labels={'log_{10}(RMSE / degree)','Mean runtime (ms)'};
    for M=1:2
        for metric=1:2
            nexttile; hold on;
            for N=unique(N_list(:)).'
                points=sortrows(rows(rows(:,1)==M & rows(:,2)==N,:),3);
                values=points(:,4+metric);
                if metric==1, values=log10(values); end
                plot(points(:,3),values,'-o','DisplayName',sprintf('N = %d',N));
            end
            grid on; xlabel('SNR (dB)'); ylabel(labels{metric});
            title(sprintf('%d source(s)',M)); legend('show','Location','best');
        end
    end
    exportgraphics(fig,fullfile(output_dir,'rmse_runtime.png'),'Resolution',200);
    exportgraphics(fig,fullfile(output_dir,'rmse_runtime.pdf'),'ContentType','vector');
    fprintf('Results and plots: %s\n',output_dir);
end

function [Y,truth]=observation(seed,M,z,k,snr_db)
    rng(seed,'twister');
    if M==1
        truth=randi([-60,60]);
        clean=exp(1j .* k .* z .* sind(truth)) .* randn(1,1);
        power=mean(abs(clean).^2,'all');
    else
        truth=[-2.5,2.5];
        raw=(randn(2,32)+1j*randn(2,32))/sqrt(2);
        covariance=(raw*raw')/32; covariance=(covariance+covariance')/2;
        [vectors,values]=eig(covariance);
        inverse_root=vectors*diag(1./sqrt(real(diag(values))))*vectors';
        sources=(inverse_root*raw)./sqrt(2);
        clean=exp(1j .* k .* z .* sind(truth))*sources;
        power=1;
    end
    noise_variance=power .* 10.^(-snr_db./10);
    Y=clean+sqrt(noise_variance./2).*(randn(size(clean))+1j.*randn(size(clean)));
end
