classdef arblayerfullconn < handle
    % Arbitrary layer fully connected feedfoward + backpropagation
    
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
        momentum = 0.0; % Default 0.9
        
        % RMSprop
        decay = 0.9; % Default 0.9
        eps = 1e-8; % Default 1e-8
    end
    
    methods
        %%
        function obj = arblayerfullconn(net_size, array_interface, varargin)
            % Declare multiple layer fully connected network
            
            okargs = {'optimizer', 'software'};
            defaults = {'SGD', false}; % Minimum batch_size as default.
            [obj.optimizer, obj.software] = internal.stats.parseArgs(okargs,defaults,varargin{:});
            
            % Dimension of the network
            obj.net_size=net_size; % net_size: column vector, e.g. [4;64;2]
            
            % Link with array object
            obj.array_interface = array_interface;
                        
            % Assign dimension of some class properties
            obj.weights=cell(length(obj.net_size)-1,1);
            obj.biases=obj.weights;
            obj.last_nabla_w=obj.weights;
            obj.last_nabla_b=obj.weights;
            obj.z=obj.weights;
            obj.a=obj.weights;
            obj.nabla_w=obj.weights;
            obj.nabla_b=obj.weights;
            obj.grad_mean_sqr_w=obj.weights;
            obj.grad_mean_sqr_b=obj.weights;
            
            % Initialization weight and bias
			
            % Initalize weight (experimental only)
            if ~obj.software 
                obj.array_interface.initialize_weights;
            end
			
            for i=1:(length(net_size)-1)
                
                % Read initial weight weights
                if ~obj.software
                    obj.weights{i} = obj.array_interface.read_weights(i);
                else
                    obj.weights{i} = 1*(rand(net_size(i+1),net_size(i))-0.5);
                end
                
                % ---------------------------------------------------------
                % Bias initialization (multiple methods here)
                % ---------------------------------------------------------
                obj.biases{i} = 1*(rand(net_size(i+1), 1)-0.5);
                % obj.biases{i} = ones(net_size(i+1), 1); % Constant 1
                
                % Momentum history initialization
                obj.last_nabla_w{i}=0;
                obj.last_nabla_b{i}=0;
                
                % RMSprop initialization
                if strcmp(obj.optimizer, 'RMSprop')
                    obj.grad_mean_sqr_w{i}=0;
                    obj.grad_mean_sqr_b{i}=0;
                end
            end
        end
        %%
        function output=forwardpass(obj,input,length_DPE_data_keep)
            % Forward pass
            % input: cell column vector (each cell a 4x1 double vector)
            
            % Convert cell vector to matrix
            input_matrix=cell2mat(input');
            
            for i=1:(length(obj.net_size)-1)
                
                % DPE
                if obj.software % Software
                    temp_dpe = obj.weights{i}*input_matrix;
                else % Physical/simu array
                    temp_dpe = obj.array_interface.dot_product(i,input_matrix);
                end
                
                temp_z = temp_dpe+repmat(obj.biases{i},1,length(input));
                
                % Keep those DPE for minibatch inputs (for backpropagation)
                obj.z{i}=temp_z(:,1:length_DPE_data_keep);
                
                % Activation
                if i<(length(obj.net_size)-1)
                    temp_a = obj.relu(temp_z); % Most layer: ReLU
                    input_matrix = temp_a; % Next input = activation now
                else
                    temp_a = temp_z; % Last layer: linear
                    output=temp_a; % Output
                end
                
                % Keep those DPE for minibatch
                obj.a{i}=temp_a(:,1:length_DPE_data_keep); 
            end
        end
        %%
        function MSE_loss=update_mini_batch(obj, input, label)
            % Update weights/biases to a single minibatch.
            % Each column of 'input' is an input, same for 'label'.
            
            % Initalize the variables to accumulate gradients
            for i=1:(length(obj.net_size)-1)
                obj.nabla_b{i} = zeros(size(obj.biases{i}));
                obj.nabla_w{i} = zeros(size(obj.weights{i}));
            end
            
            % Minibatch size
            [~,size_minibatch] = size(input);
            
            % Initialize loss
            MSE_loss=0;
            
            % Backpropagate over minibatch
            for i=1:size_minibatch
                % Backpropagation (i is index to fetch from existing DPE)
                MSE_loss=MSE_loss+obj.backpropagation(input(:,i),label(:,i),i);
            end
            MSE_loss=MSE_loss/2/size_minibatch; % Loss=1/2N*sum(singleloss)
            
            % Update weights and biases
            for i=1:(length(obj.net_size)-1)
                
                % Devide by minibatch size
                obj.nabla_w{i}=1/size_minibatch*obj.nabla_w{i};
                obj.nabla_b{i}=1/size_minibatch*obj.nabla_b{i};
                
                if strcmp(obj.optimizer,'SGD')
                    % With momentum
                    obj.nabla_w{i}=-obj.learningrate*obj.nabla_w{i}+obj.momentum*obj.last_nabla_w{i};
                    obj.nabla_b{i}=-obj.learningrate*obj.nabla_b{i}+obj.momentum*obj.last_nabla_b{i};
                elseif strcmp(obj.optimizer,'RMSprop')
                    % With momentum
                    obj.grad_mean_sqr_w{i}=obj.decay*obj.grad_mean_sqr_w{i}+(1-obj.decay)*obj.nabla_w{i}.^2;
                    obj.grad_mean_sqr_b{i}=obj.decay*obj.grad_mean_sqr_b{i}+(1-obj.decay)*obj.nabla_b{i}.^2;
                    obj.nabla_w{i}=-obj.learningrate*obj.nabla_w{i}./(obj.grad_mean_sqr_w{i}.^0.5+obj.eps)...
                        +obj.momentum*obj.last_nabla_w{i};
                    obj.nabla_b{i}=-obj.learningrate*obj.nabla_b{i}./(obj.grad_mean_sqr_b{i}.^0.5+obj.eps)...
                        +obj.momentum*obj.last_nabla_b{i};
                else
                    error('Optimizer not supported yet.');
                end
                
                % =========================================================
                % (Regularization) Make the mean nabla_w, nabla_b zero
                % =========================================================
                obj.nabla_w{i}=obj.nabla_w{i}-mean(obj.nabla_w{i}(:));
                obj.nabla_b{i}=obj.nabla_b{i}-mean(obj.nabla_b{i}(:));               
                
                % (Momentum) remember the recent nabla_w / _b
                obj.last_nabla_w{i}=obj.nabla_w{i};
                obj.last_nabla_b{i}=obj.nabla_b{i};
                
                % Update biases
                temp = obj.biases{i} + obj.nabla_b{i};
                temp(temp>2.5)=2.5; % Safe upper limit
                temp(temp<-2.5)=-2.5; % Safe lower limit
                obj.biases{i}=temp;
                
                % Software weight update
                if obj.software
                    obj.weights{i}=obj.weights{i}+obj.nabla_w{i};
                end
            end
            
            % Hardware weight update
            if ~obj.software
                % Phyiscal update memristor G together
                obj.array_interface.set_weights_gradient(obj.nabla_w);
            
                % Read new physical weights
                for i=1:(length(obj.net_size)-1)              
                    obj.weights{i} = obj.array_interface.read_weights(i);
                end
            end
        end
        %%
        function single_loss=backpropagation(obj, input_single, label_single, sample_index)
            % Backpropagation of a single sample
            % input_single/label_single: column vector
            
            % Forward pass (output_single: column vector) already done
            % Fetch dpe output at obj.stat(2).a(:,i)
            output_single=obj.a{length(obj.net_size)-1}(:,sample_index);
            
            % Single sample loss
            single_loss=norm(label_single-output_single,2);
            
            % Backward pass
            for i = (length(obj.net_size)-1):-1:1
                
                % dErr/dZ of current layer
                if i == (length(obj.net_size)-1)
                    % Last layer: linear activation
                    dErr_dz = obj.cost_derivative(output_single, label_single); 
                else
                    % Rest layers
                    dErr_dz = (obj.weights{i+1}' * dErr_dz).*...
                        obj.relu_derivative(obj.z{i}(:,sample_index));
                end
                
                % dErr/db of current layer
                dErr_db=dErr_dz;
                
                % dErr/dw of current layer
                if i>1
                    dErr_dw=dErr_dz*obj.a{i-1}(:,sample_index)'; % Rest layer
                else
                    dErr_dw=dErr_dz*input_single'; % First layer
                end
                
                % Accumulate dErr/db and dErr/dw (over minibatch)
                obj.nabla_b{i} = obj.nabla_b{i} + dErr_db;
                obj.nabla_w{i} = obj.nabla_w{i} + dErr_dw;
                
            end
        end
    end
    methods(Static)
        %%
        function costderi_output=cost_derivative(output,label)
            % The error is defined as MSE here.
            % The last layer activation is linear.
            % Output/label/costderi_output: a column vector
            
            costderi_output=-(label-output);
        end
        %%
        function relu_output=relu(relu_inputs)
            % ReLU
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