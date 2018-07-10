clear all; close all;

% Constant
MAX_EPISODE = 300; WIN_CRITERIA = 1000; NET_SIZE = [4;48;48;2];
NET_SIZE_length = length(NET_SIZE)-1;
RATIO_WEIGHT_CONDUCTANCE = 0.00025*ones(NET_SIZE_length,1); % 0.00005
OPTIMIZER = 'RMSprop'; SOFTWARE = false;

%% Array obj and Array Interface obj

%update_fun=@(G,Vm,Vt) (95e-6+10e-6*rand(size(G))-G).*(Vm<0)+...                % RESET part
%    ((Vt-0.7).*(Vt>0.5)*98e-6+95e-6+10e-6*rand(size(G))-G).*(Vm>0).*(Vt>0.1);  % SET part
%base = multi_array(sim_array1({'random' [128 64] 50e-6 100e-6},update_fun,0, Inf));

load('Apr_9_good_row_col');
base = multi_array(real_array2(good_row, good_col));
base.add_sub([20 1], [8 48]);
base.add_sub([29 1], [96 48]);
base.add_sub([20 49], [48 4]);

% Declare array interface obj
array_interface = arrayinterface_v2(base, NET_SIZE, RATIO_WEIGHT_CONDUCTANCE);

%% RNG seed
rng(5);

%% Enviroment and Agent

env=CartPole; % Enviroment
agent=Agent(NET_SIZE, array_interface, 'optimizer', OPTIMIZER, 'software', SOFTWARE); % Agent

%% Data saving

% Including hardware information
s_hist=struct('epi_num',[],'state',[],'action',[],'weights',[],'bias',[],...
    'nabla_b',[],'nabla_w',[],'v_gate',[],'G_full',[],'I_dpe',[]);
s_hist_pointer=1;

% Performance and loss (independent vectors)
perf = NaN(MAX_EPISODE, 1);
MSE_loss = NaN(MAX_EPISODE*50,1);

%% Plot

% Initialize plot
h=figure(1);
set(h,'name','RL','numbertitle','off','Units', 'normalized', 'Position', [0,0,1,1]);

perf_panel=subplot(3,NET_SIZE_length+1,1);
loss_panel=subplot(3,NET_SIZE_length+1,NET_SIZE_length+2);
I_raw_panel=subplot(3,NET_SIZE_length+1,2*NET_SIZE_length+3);

for i=1:NET_SIZE_length 
    G_full_panel(i)=subplot(3,NET_SIZE_length+1,1+i);
    v_gate_panel(i)=subplot(3,NET_SIZE_length+1,NET_SIZE_length+2+i);
    weights_panel(i)=subplot(3,NET_SIZE_length+1,NET_SIZE_length*2+3+i);
end

%%
for episode_counter=1:MAX_EPISODE   
    
    % Start a new episode
    stat=env.reset_episode; % Get a new state
    episode_over_flag=0; % Episode end flag (false)
    reward_episode=0; % Total reward of episode (zero)
    
    % Each episode
    while ~episode_over_flag
        
        % The first time no replay (nothing in memory)
        if s_hist_pointer > 1
            % Get experience from memory and train
            % Also do DPE for action prediction
            MSE_loss(s_hist_pointer) = agent.replay_predict(stat);

            % Record
            s_hist(s_hist_pointer).G_full = array_interface.G_array; % The fast read G of the entire chip (read after last time update)
            s_hist(s_hist_pointer).weights = agent.brain.net.weights; % Weights
            s_hist(s_hist_pointer).bias = agent.brain.net.biases; % Biases
            s_hist(s_hist_pointer).v_gate = array_interface.V_gate_last; % Gate voltages
            s_hist(s_hist_pointer).nabla_w = agent.brain.net.nabla_w; % Gradients (weights)
            s_hist(s_hist_pointer).nabla_b = agent.brain.net.nabla_b; % Gradients (biases)
            s_hist(s_hist_pointer).I_dpe = array_interface.I_raw; % Raw DPE currents
        end
        
        % The agent decides what action to take
        % Physical DPE done in replay
        a=agent.act;
        
        % Display
        display(['Step=',num2str(s_hist_pointer),' Reward=',num2str(reward_episode),' Loss=',num2str(MSE_loss(s_hist_pointer))]);

        % Performs this action, get s_ (new state) and r (a reward)
        [stat_,r,episode_over_flag]=env.nextstate(a);
        
        % Record
        s_hist(s_hist_pointer).epi_num = episode_counter; % Episode number
        s_hist(s_hist_pointer).state = stat; % Current state
        s_hist(s_hist_pointer).action = a; % Current action
        
        % Next state is NaN (as a mark for Bellman equation) if game over
        if episode_over_flag
            stat_ = NaN;
        end

        % Add to memory (as a 1x4 cell array)
        agent.observe({stat,a,r,stat_});
        
        % Cummulated reward
        reward_episode=reward_episode+r;
        
        % State evole
        stat=stat_;
        
        % Plot perf/weights of this episode
        if and(~mod(s_hist_pointer, 250), s_hist_pointer~=0)
            plot_perf_weight(perf_panel,loss_panel,I_raw_panel,...
                G_full_panel,v_gate_panel,weights_panel,...
                perf(1:episode_counter),MSE_loss(1:s_hist_pointer),s_hist(s_hist_pointer).I_dpe,...
                s_hist(s_hist_pointer).G_full,s_hist(s_hist_pointer).weights,s_hist(s_hist_pointer).v_gate);
        end
        
        % Increase the pointer
        s_hist_pointer = s_hist_pointer+1;

    end
    
    % Performance of this episode
    perf(episode_counter)=reward_episode;
    
    % Exit condition (performance reached the threshold)
    if reward_episode > WIN_CRITERIA
        plot_perf_weight(perf_panel,loss_panel,I_raw_panel,...
            G_full_panel,v_gate_panel,weights_panel,...
            perf(1:episode_counter),MSE_loss(1:s_hist_pointer),s_hist(s_hist_pointer).I_dpe,...
            s_hist(s_hist_pointer).G_full,s_hist(s_hist_pointer).weights,s_hist(s_hist_pointer).v_gate);
        break
    end
end

%% Data Save
t_now=datestr(now,'yyyymmdd HHMMSS');
f_name1=[t_now '_cartpole.mat'];
save(f_name1,'perf','MSE_loss','s_hist','-v7.3');

%% Plot function
function plot_perf_weight(perf_axes, loss_axes, I_raw_axes, ...
    G_full_axes, weights_axes, v_gate_axes,...
    perf, MSE_loss, I_dpe,...
    G_full, weights, v_gate)
    % Plot (performance, loss, voltages, and weights) per episode

    % Plot performance
    axes(perf_axes);
    plot(perf,'LineWidth',2); xlabel('Episode'); ylabel('Rewards'); title('Performance');
    
    % Plot loss function
    axes(loss_axes);
    plot(MSE_loss,'LineWidth',2); xlabel('No. Replays'); ylabel('MSE Loss'); title('Loss Fun');
    
    % Plot Raw DPE currents
    axes(I_raw_axes); hold on;
    for i=1:length(I_dpe)
        plot(I_dpe{i});
    end
    hold off; xlim([0 64]); title('Raw current');
    
    % Plot gate conductance, weights, gate voltage
    for i=1:length(weights)
        axes(G_full_axes(i)); imagesc(G_full{i},[0 5e-4]); colorbar; title(['G ' num2str(i)]);
        axes(weights_axes(i)); imagesc(weights{i}); colorbar; title(['Weight ' num2str(i)]);
        axes(v_gate_axes(i)); imagesc(v_gate{i},[0.5 1.5]); colorbar; title(['Vgate ' num2str(i)]);
    end
    
    % Draw
    drawnow
end
