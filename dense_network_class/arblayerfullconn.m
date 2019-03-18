classdef arblayerfullconn < handle
    % The class of a simple multilayer perceptron (with backpropagation)
    % Optimizer : SGD, RMSprop
    % Backend: software, array obj
    % Activation (fixed): ReLU and linear (last layer)
    % Loss (fixed): MSE

    properties
        
        % Dimension of the network
        net_size
        
        % Optimizer
        optimizer
        
        % Weights and biases
        weights
        biases
        z
        a
        nabla_w
        nabla_b
        last_nabla_w
        last_nabla_b
        grad_mean_sqr_w
        grad_mean_sqr_b
        
        % Software or hardware
        software
        
        % Learning rate
        learningrate=0.0005;
        
        % Physical/simu array
        array_interface
        
        % Momentum
        momentum = 0.0;
        
        % RMSprop
        decay = 0.9;
        eps = 1e-8;
    end
    
    methods
        %%
        function obj = arblayerfullconn(net_size, array_interface, varargin)
            % Create the multilayer perceptron obj
            
            okargs = {'optimizer', 'software'};
            defaults = {'SGD', false}; % Minimum batch_size as default.
            [obj.optimizer, obj.software] = internal.stats.parseArgs(okargs,defaults,varargin{:});
            
            % Dimension of the network
            obj.net_size = net_size; % net_size: column vector, e.g. [4;64;2]
            
            % Link with array object
            obj.array_interface = array_interface;
                        
            % Assign dimensions to properties of the class
            obj.weights = cell(length(obj.net_size)-1,1);
            obj.biases = obj.weights;
            obj.last_nabla_w = obj.weights;
            obj.last_nabla_b = obj.weights;
            obj.z = obj.weights;
            obj.a = obj.weights;
            obj.nabla_w = obj.weights;
            obj.nabla_b = obj.weights;
            obj.grad_mean_sqr_w = obj.weights;
            obj.grad_mean_sqr_b = obj.weights;
            
            % Initialization: weight and bias
			
            % Initalize weight (non-software only)
            if ~obj.software 
                obj.array_interface.initialize_weights;
            end
			
            for i=1:(length(net_size)-1)
                
                % Read initialized weights
                if ~obj.software
                    obj.weights{i} = obj.array_interface.read_weights(i);
                else
                    obj.weights{i} = 1*(rand(net_size(i+1),net_size(i))-0.5);
                end
                
                % Bias initialization (multiple methods here)                
                obj.biases{i} = 1*(rand(net_size(i+1), 1)-0.5);                
                
                % Momentum history initialization
                obj.last_nabla_w{i} = 0;
                obj.last_nabla_b{i} = 0;
                
                % RMSprop optimizer initialization
                if strcmp(obj.optimizer, 'RMSprop')
                    obj.grad_mean_sqr_w{i} = 0;
                    obj.grad_mean_sqr_b{i} = 0;
                end
            end
        end
        %%
        function output = forwardpass(obj, input, length_DPE_data_keep)
            % Forward pass
            % input: cell column vector
            
            % Convert cell vector to matrix
            input_matrix = cell2mat(input');
            
            for i=1:(length(obj.net_size)-1)
                
                % Matrix-vector multiplications
                if obj.software % software backend
                    temp_dpe = obj.weights{i} * input_matrix;
                else % array obj
                    temp_dpe = obj.array_interface.dot_product(i, input_matrix);
                end
                
                temp_z = temp_dpe + repmat(obj.biases{i}, 1, length(input));
                
                % Keep results for backpropagation
                obj.z{i}=temp_z(:, 1:length_DPE_data_keep);
                
                % Activation
                if i < (length(obj.net_size) - 1)
                    temp_a = obj.relu(temp_z); % ReLU
                    input_matrix = temp_a; % Next input = activation now
                else
                    temp_a = temp_z; % Linear (last layer)
                    output = temp_a; % Output
                end
                
                % Keep those DPE for minibatch
                obj.a{i} = temp_a(:, 1:length_DPE_data_keep); 
            end
        end
        %%
        function MSE_loss = update_mini_batch(obj, input, label)
            % Update weights/biases based on a minibatch
            
            % Initalize the variables to accumulate gradients
            for i = 1:(length(obj.net_size) - 1)
                obj.nabla_b{i} = zeros(size(obj.biases{i}));
                obj.nabla_w{i} = zeros(size(obj.weights{i}));
            end
            
            % Minibatch size
            [~, size_minibatch] = size(input);
            
            % Initialize loss
            MSE_loss = 0;
            
            % Backpropagate over minibatch
            for i = 1:size_minibatch
                % Backpropagation
                MSE_loss = MSE_loss + obj.backpropagation(input(:, i), label(:, i), i);
            end
            MSE_loss = MSE_loss / 2 / size_minibatch;
            
            % Update weight and biase
            for i = 1:(length(obj.net_size) - 1)
                
                % Devide by minibatch size
                obj.nabla_w{i} = 1 / size_minibatch * obj.nabla_w{i};
                obj.nabla_b{i} = 1 / size_minibatch * obj.nabla_b{i};
                
                if strcmp(obj.optimizer,'SGD')
                    % SGD optimizer
                    obj.nabla_w{i} = -obj.learningrate * obj.nabla_w{i} + ...
                        obj.momentum * obj.last_nabla_w{i};
                    obj.nabla_b{i} = -obj.learningrate * obj.nabla_b{i} + ...
                        obj.momentum * obj.last_nabla_b{i};
                elseif strcmp(obj.optimizer,'RMSprop')
                    % RMSprop optimizer
                    obj.grad_mean_sqr_w{i} = obj.decay * obj.grad_mean_sqr_w{i}...
                        + (1 - obj.decay) * obj.nabla_w{i}.^2;
                    obj.grad_mean_sqr_b{i} = obj.decay * obj.grad_mean_sqr_b{i}...
                        + (1 - obj.decay) * obj.nabla_b{i}.^2;
                    obj.nabla_w{i} = -obj.learningrate * obj.nabla_w{i}./(obj.grad_mean_sqr_w{i}.^0.5 + obj.eps)...
                        + obj.momentum * obj.last_nabla_w{i};
                    obj.nabla_b{i} = -obj.learningrate * obj.nabla_b{i}./(obj.grad_mean_sqr_b{i}.^0.5 + obj.eps)...
                        + obj.momentum * obj.last_nabla_b{i};
                else
                    error('Optimizer not supported yet.');
                end
                
                % ---------------------------------------------------------
                % Note: A trick to overcome the possible divergence at the
                % onset of the training with physical memristors, but 
                % degrades performance
                % obj.nabla_w{i} = obj.nabla_w{i} - mean(obj.nabla_w{i}(:));
                % obj.nabla_b{i} = obj.nabla_b{i} - mean(obj.nabla_b{i}(:));               
                % ---------------------------------------------------------
                
                % Save the recent nabla_w and nabla_b
                obj.last_nabla_w{i} = obj.nabla_w{i};
                obj.last_nabla_b{i} = obj.nabla_b{i};
                
                % Update biases with bounds (Note the bounds helps training
                % with physical memristors only)
                temp = obj.biases{i} + obj.nabla_b{i};
                temp(temp > 2.5) = 2.5; % Upper limit
                temp(temp < -2.5) = -2.5; % Lower limit
                obj.biases{i} = temp;
                
                % Software backend weight update
                if obj.software
                    obj.weights{i} = obj.weights{i} + obj.nabla_w{i};
                end
            end
            
            % Hardware weight update
            if ~obj.software
                % Phyiscal update memristors of all layers together
                obj.array_interface.set_weights_gradient(obj.nabla_w);
            
                % Read new physical weights
                for i = 1:(length(obj.net_size) - 1)
                    obj.weights{i} = obj.array_interface.read_weights(i);
                end
            end
        end
        %%
        function single_loss=backpropagation(obj, input_single, label_single, sample_index)
            % Backpropagation of a single sample
            
            % Fetch forward pass output
            output_single = obj.a{length(obj.net_size) - 1}(:, sample_index);
            
            % Single sample loss
            single_loss = norm(label_single - output_single, 2);
            
            % Backward pass
            for i = (length(obj.net_size) - 1):-1:1
                
                % dE/dZ of current layer
                if i == (length(obj.net_size)-1)
                    % Last layer: linear activation
                    dErr_dz = obj.cost_derivative(output_single, label_single); 
                else
                    % Rest layers: ReLU
                    dErr_dz = (obj.weights{i+1}' * dErr_dz).*...
                        obj.relu_derivative(obj.z{i}(:, sample_index));
                end
                
                % dErr/db of current layer
                dErr_db = dErr_dz;
                
                % dErr/dw of current layer
                if i > 1
                    dErr_dw = dErr_dz * obj.a{i - 1}(:, sample_index)'; % Rest layer
                else
                    dErr_dw = dErr_dz * input_single'; % First layer
                end
                
                % Accumulate dE/db and dE/dw over minibatch
                obj.nabla_b{i} = obj.nabla_b{i} + dErr_db;
                obj.nabla_w{i} = obj.nabla_w{i} + dErr_dw;
                
            end
        end
    end
    methods(Static)
        %%
        function costderi_output=cost_derivative(output,label)
            % Delta of the output neuron (dE/d(output))
            % MSE error assumed
            % Output/label/costderi_output: a column vector
            
            costderi_output = -(label - output);
        end
        %%
        function relu_output=relu(relu_inputs)
            % ReLU function
            % relu_inputs/relu_outputs: matrix (each column one input)
            
            relu_output=relu_inputs.*(relu_inputs>0);
        end
        %%
        function reluderi_output = relu_derivative(reluderi_input)
            % ReLU derivative
            % reluderi_input/reluderi_output: a column vector
            
            reluderi_output = reluderi_input>0;
        end
    end
end
