classdef Agent < handle
    % Agent (contains brain and memory)
    
    properties
        
        memory_capacity=20000; % Was 100000 by default
        % Each time recall memory size
        batch_size=80; % Was 64 by default
        
        % Discount factor
        gamma=0.99
        
        % Dynamic exploration limits
        max_epsilon=1;
        min_epsilon=0.01;
        lambda=0.001; % speed of decay. Was 0.001 by default
        
        % Input and Output size of Q net
        stateCnt
        actionCnt
        
        % Interface with other classes
        brain
        memory
        
        % Exploration probability
        epsilon
        steps % Truly total steps, determining epsilon
        
        % Store the DPE result of stat_, for act prediction
        stat_predict
        
    end
    
    methods
        %%
        function obj = Agent(net_size, array_interface, varargin)
            % Agent declaration
            
            % Input (state vector) size
            obj.stateCnt = net_size(1);
            
            % Output (SET of action) size
            obj.actionCnt = net_size(end);
            
            % Interface with brain and memory classes
            obj.brain=Brain(net_size, array_interface, varargin{:});
            obj.memory=Memory(obj.memory_capacity);
            
            % Initial episilon
            obj.epsilon=obj.max_epsilon;
            obj.steps=0;
            
            % The very first DPE result of stat (nothing)
            obj.stat_predict=NaN;
        end
        %%        
        function action = act(obj)
            % Suggesting an action based on current state, stat 4x1 double
            
            % Random explore (probability, or no prediction available round)
            if or(rand<obj.epsilon, isnan(obj.stat_predict))
                % Random exploration
                action=-1+2*(rand>0.5); % Either -1 or 1
            else
                % Greedy policy from Q-network
                [~,action]=max(obj.stat_predict);
                if action==1
                    action=-1; % N1 win: action -1
                else
                    action=1; % N2 win: action 1
                end
            end
        end
        %%        
        function observe(obj, new_sample)
            % Add observation to memory & change exploration probability
            % new_sample should be a single cell
            
            % Add {state, action, reward, next_state} to memory
            obj.memory.add_memory(new_sample);
            
            % The time dependent probability of exploration
            obj.steps=obj.steps+1;
            obj.epsilon=obj.min_epsilon+(obj.max_epsilon-obj.min_epsilon)*exp(-obj.lambda*obj.steps);
        end
        %%
        function MSE_loss = replay_predict(obj, stat_to_pred)
            % (1) Replay and learn from experience
            % (2) Also helps to do DPE for action prediction (stat_to_pred)
            
            % Fetch a batch of samples from memory (each row: 1x4 cell)
            minibatch=obj.memory.sample_memory(obj.batch_size);
            
            % Verify the batch size (it may < obj.batch_size)
            batchlen=size(minibatch,1);
            
            % States and next_states of the minibatch
            states=minibatch(1:batchlen,1);
            % Next states (of all samples in the minibatch)
            states_=minibatch(1:batchlen,4);
			
			%--------------------------------------------------------------------------------
            % Wrap stat_to_pred (4x1 double) to a cell
            stat_to_pred_cell{1,1}=stat_to_pred;

            % === Physical DPE here only ===
            % 'Inference' with 3 cell column vectors:
            % (1) current states (2) next states (3) stat_pred 
            % Note only (2) may contain NaN cells.
            % The length(state) will be used to keep relavant information for back propagation
            temp = obj.brain.predict([states; states_; stat_to_pred_cell;], length(states));
            
            % Allocate DPE results to (1)(2)(3)
            % Each result is one or few column vectors
            p=temp(:,1:length(states));
            p_=temp(:,length(states)+1:length(states)+length(states_));
            obj.stat_predict=temp(:,end);
     		%--------------------------------------------------------------------------------
            
            % Prepare the batch 'Inputs' and 'Labels' for training
            train_input=zeros(obj.stateCnt, batchlen);
            train_label=zeros(obj.actionCnt, batchlen);
            
            for i=1:batchlen

                % Pick each sample from the batch
                s = minibatch{i,1}; % Get state
                a = minibatch{i,2}; % action
                r = minibatch{i,3}; % reward
                s_ = minibatch{i,4}; % next_state
            
                % Initial 'label' with current inference of (s), 2x1 vector
                t = p(:,i);

                if isnan(s_)
                    % If finished, the 'label' for 'a' neuron is r.
                    t(1+(a>0)) = r; % or r!!!
                else
                    % Else is r+GAMMA* the maximum neuron output of (s_)
                    t(1+(a>0)) = r + obj.gamma * max(p_(:,i));
                end

                % Assign "Input" and "Label"
                train_input(:,i) = s;
                train_label(:,i) = t;
            end
            
            % Train it
            MSE_loss = obj.brain.train(train_input, train_label);
        end
    end
end