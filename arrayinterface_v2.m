classdef arrayinterface_v2 < handle
% Interface between network objects & array objects

    properties
        
        base
        net_size
        diff_pair_ori
        weight_scaling
        
        % Blind update parameters
        th_set = 0;
        th_reset = -0; % Should be negative
        
        Vg_max = 1.5; % Max SET gate voltage
        Vg_min = 0.6; % Min SET gate voltage
        V_set = 2.5; % Fixed SET voltage
        V_reset = 1.7; % Fixed RESET voltage
        V_gate_reset = 5; % Fixed RESET gate voltage
        
        % Recording information
        V_gate_last
        G_array
        G_full
        I_raw
        
        % Read parameters
        V_read = 0.2;       
                
        % Delta_conductancd / Delta_V_gate (98e-6/(1.5V-0.8V))?
        ratio_G_Vg = 125e-6;  % Was 100e-6 in the 1st try

    end
    
    methods
        %%
        function obj = arrayinterface_v2(base,net_size, weight_scaling)
            % Construct interface
            % Array: (? x 1 array of memristor array objects)
            % net_size: (? x 1 array of layers, including input neurons)
            
            obj.base = base;
            obj.net_size = net_size;
            
            % Orientation of the differential pair
            obj.diff_pair_ori = NaN(length(net_size)-1,1); 
            
            % Fixed weight scaling factors (weight * scaling = G)
            if length(net_size)==length(weight_scaling)+1
                obj.weight_scaling = weight_scaling;
            else
                error('Weight scaling dimension mismatch')
            end
            
            % Initialize property: G_array
            obj.G_array=cell(length(net_size)-1,1);
            
            % Orientation of differential pair
            for i=1:(length(net_size)-1)
                
                % Initial random weight ===================================
                weight_to_write = 0.5*(rand(net_size(i),net_size(i+1))-0.5);
                weight_pos=weight_to_write.*(weight_to_write>=0);
                weight_neg=-weight_to_write.*(weight_to_write<0);
                
                if isequal(obj.base.subs{i}.net_size, [2*net_size(i), net_size(i+1)])
                    % 1 for Vertical differential pair
                    obj.diff_pair_ori(i)=1;
                    % Combine pos/neg weight to G matrix
                    temp = NaN([2*net_size(i), net_size(i+1)]);
                    temp(1:2:end-1,:) = weight_pos;
                    temp(2:2:end,:) = weight_neg;
                    % Random initial weight (gate voltage)
                    temp = 0.5*(obj.Vg_min+obj.Vg_max)+temp*obj.weight_scaling(i)/obj.ratio_G_Vg;
                elseif isequal(obj.base.subs{i}.net_size, [net_size(i), 2*net_size(i+1)])
                    % 2 for Horizontal differential pair
                    obj.diff_pair_ori(i)=2;
                    % Combine pos/neg weight to G matrix
                    temp = NaN([net_size(i), 2*net_size(i+1)]);
                    temp(:,1:2:end-1) = weight_pos;
                    temp(:,2:2:end) = weight_neg;
                    % Random initial weight (gate voltage)
                    temp = 0.5*(obj.Vg_min+obj.Vg_max)+temp*obj.weight_scaling(i)/obj.ratio_G_Vg;
                else
                    error('Diff pair is neither Vertical or Horizonal');
                end
                 
                temp(temp>obj.Vg_max)=obj.Vg_max; % Enforce Vg upper bound
                temp(temp<obj.Vg_min)=obj.Vg_min; % Enforce Vg lower bound
                obj.V_gate_last{i}=temp;
            end
        end
        %%
        function initialize_weights(obj)
            % Initailzie weights by RESET them
            
            % RESET the entire arrays
            Vr=cell(length(obj.net_size)-1,1); Vr(:)={obj.V_reset};
            obj.base.update_subs('GND', Vr, obj.V_gate_reset); % RESET all
            obj.base.update_subs(obj.V_set, 'GND', obj.V_gate_last); % SET those of the array
            
            % Save the fast read conductance
            [obj.G_array, obj.G_full] = obj.base.read_subs('mode', 'fast');
        end
        %%
        function w = read_weights(obj, layer)
            % Read weights
            
            % Read weight from the property
            G_read = obj.G_array{layer};
            
            % Differential pair
            if obj.diff_pair_ori(layer) == 1
                % Vertical pair
                G = G_read(1:2:end-1,:)-G_read(2:2:end,:);
            else
                % Horizontal pair
                G = G_read(:,1:2:end-1)-G_read(:,2:2:end);
            end
            
            % Scaling back (from G to w)
            w = G/obj.weight_scaling(layer);
            
            % Transpose
            w=w';
        end
        %%
        function dpe_result = dot_product(obj, layer, input, varargin)
            % Vector-Matrix multiplication
            % input: matrix
            
            % Voltage scaling (input * scaling = voltage)            
            voltage_scaling = obj.V_read./max(abs(input));
            voltage_scaling(~any(input))=1; % Case the whole input is '000...000'
            voltage_scaling_matrix=repmat(voltage_scaling,size(input,1),1);
            
            
            % Differential pair
            if obj.diff_pair_ori(layer) == 1
                % Vertical pair case
                V_input=NaN(size(input).*[2 1]);
                V_input(1:2:end-1,:)=input.*voltage_scaling_matrix;
                V_input(2:2:end,:)=-V_input(1:2:end-1,:);
                % Each column is an individual output current
				if layer==1 % Layer 1: Limited number of rows
	                I_output = obj.base.subs{layer}.read_current(V_input, 'gain', 2);
                else % Layer 2,3: More rows
					I_output = obj.base.subs{layer}.read_current(V_input);
		 		end
                obj.I_raw{layer}=I_output;
            else
                % Horizontal pair
                V_input = input.*voltage_scaling_matrix;
                % Each column is an individual output current
                I_temp = obj.base.subs{layer}.read_current(V_input);
                I_output=I_temp(1:2:end-1,:) - I_temp(2:2:end,:);
                obj.I_raw{layer}=I_temp;
            end
            
            % Scaling back (voltage and weight scaling)
            dpe_result=I_output./voltage_scaling/obj.weight_scaling(layer);
            
        end
        %%
        function set_weights_gradient(obj, grad)
            % Blind weight update
            
            % Initialize reset, gate, gate apply cell arrays.
            Vr=cell(length(grad),1);
            Vgs=Vr;
            Vgs_apply=Vr;
            
            % Set voltages of each layer
            for layer=1:length(grad)
                
                % Transpose and negative (weight = weight - grad * l_rate)
                grad_temp = grad{layer}';
            
                % Gradient scaling and to voltage conversion
                dV_temp = grad_temp*obj.weight_scaling(layer)/obj.ratio_G_Vg;
            
                % Change of gate voltage
                if obj.diff_pair_ori(layer)==1
                    % Vertical case
                    dV = NaN(size(dV_temp).*[2 1]);
                    dV(1:2:end-1,:) = dV_temp; dV(2:2:end,:) = -dV_temp; 
                else
                    % Horizontal case
                    dV = NaN(size(dV_temp).*[1 2]);
                    dV(:,1:2:end-1) = dV_temp; dV(:,2:2:end) = -dV_temp; 
                end
            
                % Determine how to update:
                % RESET if dV is negative (or <th_reset)
                Vr{layer} = obj.V_reset .* (dV < obj.th_reset);
                % SET if (1) dV is postive (or > th_set) (2) Just RESET
                Vgs{layer} = obj.V_gate_last{layer} + dV.*(dV>= obj.th_set | dV<=obj.th_reset);
                % Regulate the min and max SET gate voltage
                Vgs{layer}(Vgs{layer} > obj.Vg_max) = obj.Vg_max;
                Vgs{layer}(Vgs{layer} < obj.Vg_min) = obj.Vg_min;
                % Gate voltages applied (skip those do not need changes)
                Vgs_apply{layer} = Vgs{layer} .* (dV>obj.th_set | dV<obj.th_reset);                 
            end
            
            % Update
            obj.base.update_subs('GND', Vr, obj.V_gate_reset); % RESET
            obj.base.update_subs(obj.V_set, 'GND', Vgs_apply ); % SET
            
            % Save the fast read conductance
            [obj.G_array, obj.G_full] = obj.base.read_subs('mode', 'fast');
            
            % Save updated gate voltages
            obj.V_gate_last = Vgs;
        end 
    end
end