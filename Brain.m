classdef Brain < handle
    % Brain of the Agent
    
    properties
        
        % Size of network
        net_size
        
        % Neural network
        net
    end
    
    methods
        function obj = Brain(net_size, array_interface, varargin)
            % Creat a brain
            
            obj.net_size=net_size;
            
            % Creat a two-layer feedforward fully connected network
            obj.net=arblayerfullconn(net_size, array_interface, varargin{:});
            
        end
        %%              
        function o=predict(obj, stat, length_DPE_data_keep)
            % Inference of Q-net with given stat
            % stat: cell column vectors, each cell 4x1 double vector,
            % may contain NaN cell
            
            % Flags of NaN cells
            flag_NaN = cellfun(@(x) any(isnan(x(:))), stat); % Column logic vector
            % NaN free input to DPE
            input = stat(~flag_NaN);
            
            % Inference return (each column one inference)
            temp=obj.net.forwardpass(input, length_DPE_data_keep);
            
            % Prepare output matrix by including NaNs back
            o=NaN(obj.net_size(end), length(stat));
            flag_NaN_matrix=repmat(flag_NaN',obj.net_size(end),1);
            o(~flag_NaN_matrix)=temp;
        end
        %%
        function MSE_loss=train(obj, input, label)
            % Train (one update over averaged gradients of the batch)
            
            MSE_loss=obj.net.update_mini_batch(input, label);
        end
    end
end

