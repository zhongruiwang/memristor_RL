classdef subarray1 < memristor_array
    % This class is used to access multiple subsections of a base array
    % without disconnecting and reconnecting each time. The subarray must
    % be contiguous.
    properties
        net_size
        net_corner
        array
        mask
    end
    methods
%%
        function obj = subarray1(base,net_corner,net_size)
            % OBJ = SUBARRAY1(BASE,NET_CORNER,NET_SIZE) creates a subarray
            % object on the memristor_array BASE. 
    
            % Size is n_in by n_out
            array_size = base.net_size;
            if any(net_size+net_corner-1>array_size)
                error('Network exceeds array bounds')
            end
            
            obj.net_size = net_size;             
            obj.net_corner = net_corner;
            
            % Create a logical mask showing which elements are part of the
            % net:
            obj.mask = false(array_size);
            obj.mask(net_corner(1):net_corner(1)+net_size(1)-1, net_corner(2):net_corner(2)+net_size(2)-1) = true;
            
            % Save the base array (Does this copy, or does it save a pointer?):
            obj.array = base;
        end
%%        
        function conductances = read_conductance(obj, varargin)
            % G = OBJ.READ_CONDUCTANCE() reads all conductances in the
            % array and returns them as a matrix.
            % G = OBJ.READ_CONDUCTANCE('NAME',VALUE) controls some
            % parameters of the read operation. Optional arguments and
            % default values are:
            %   'v_read' = 0.2
            %   'gain' = 2
            %   'mode' = 'slow'
            %s = 'subarray'; 
            conductances = obj.array.read_conductance(varargin{:});
            if any(size(conductances) ~= obj.net_size)
                conductances = reshape(conductances(obj.mask),obj.net_size);
            end
        end
%%        
        function update_conductance(obj, V_in, V_out, V_gate)
            % OBJ.UPDATE_CONDUCTANCE(V_IN,V_OUT,V_TRANS) sends and update
            % pulse to the array.
            % All voltages can be 'GND', a scalar, a column vector of length
            % net_size(1), or a matrix of size net_size.
            % Best practice is to set one of V_IN or V_OUT to 'GND'. If
            % both are nonzero, then two separate pulses (one SET and one
            % RESET) will be applied.
            
            obj.array.update_conductance(obj.expand(V_in),obj.expand(V_out),obj.expand(V_gate))
        end
%%        
        function [I_out,full_out] = read_current(obj,V_in, varargin)
            % I_OUT = OBJ.READ_CURRENT(V_IN) does matrix-matrix multiplication.
            % I_OUT = transpose(G)*V_IN when G is 128x64. Note that this
            % function interprets the orientations of I_OUT and V_in
            % differently than VMM_hardware.
            % V_IN can be either a row cell array of column vectors or a matrix.
            % In either case, I_OUT will be a matrix.
            % FULL_OUT is the full array's output.
            % Optional args:
            %   gain = 1
            
            if iscell(V_in)
                try
                    cell2mat(V_in);
                catch
                    error('Unrecognized input format')
                end
            end
            temp = padarray(V_in,obj.net_corner(1)-1,0,'pre');
            V_in = padarray(temp,obj.array.net_size(1)-obj.net_corner(1)-obj.net_size(1)+1,0,'post')';
            V_in = V_in';
            
            full_out = obj.array.read_current(V_in, varargin{:}); % Returns columns
            I_out = full_out(obj.net_corner(2):obj.net_corner(2)+obj.net_size(2)-1,:);
        end
          
%%        
        function c = expand(obj,a,x)
            % B = OBJ.EXPAND(A) zero-pads array A so that it lines up right
            % with the object matrix.
            % B = OBJ.EXPAND(A,X), where X is a scalar, pads with X
            % instead.
            % This is a utility designed for use inside of other methods on
            % this class.
            
            % First part: Expand a to net_size
            if all(size(a) == obj.net_size)
                b = a;
            elseif isscalar(a)
                b = repmat(a,obj.net_size);
            elseif strcmpi(a,'GND')
                b = zeros(obj.net_size);
            elseif any(size(a) == obj.net_size)
                if iscolumn(a) && length(a) == obj.net_size(1)
                    b = repmat(a,1,obj.net_size(2));
                elseif isrow(a) && length(a) == obj.net_size(2)
                    b = repmat(a,obj.net_size(1),2);
                else
                    error('Not sure how to expand this input')
                end
            else
                error('Not sure how to expand this input')
            end            
                        
            % Second part: Pad b to array_size
            if nargin<3
                x = 0;
            end
            
            c = zeros(obj.array.net_size)+x;
            c(obj.mask) = b;
        end
    end
end