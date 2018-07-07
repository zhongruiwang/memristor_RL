classdef real_array2 < memristor_array
    % This class interacts with the crossbar through DPE_WRITING(). It uses
    % only a subsection of the array, and devices outside that subsection
    % are unaffected. 
    % Unlike real_array1, this class provides the ability to choose
    % arbitary rows/columns out of the whole array. The devices used are
    % those located at the intersections of the rows and columns selected.
    % This class is intended to be fairly lightweight, with few built-in
    % functions, but it a wide range of algorithms can be written on top of
    % it. 
    % Construction syntax: 
    % OBJ = REAL_ARRAY2(ROW_SELECT, COL_SELECT,ARRAY_SIZE = [128 64])
    %
    
    
    properties
        net_size
        array
        array_size
        mask
        row_select
        col_select
        
    end
    methods
%%
        function obj = real_array2(row_sel, col_sel, array_size)
            % OBJ = REAL_ARRAY2(ROW_SELECT, COL_SELECT,ARRAY_SIZE = [128 64])
            % creates a memristor_array using the row-column intersections
            % specified by ROW_SELECT, COL_SELECT. Repeated rows/columns
            % are ignored, and the row and column indices are sorted so
            % that the physical ordering of the devices matches their
            % indices in the class.
            % This class interacts with the crossbar through DPE_WRITING.
    
            % Size is n_in by n_out
            obj.array_size = [128 64];
            
            if nargin == 3
                obj.array_size = array_size;
            end
            
            if ~isvector(row_sel) || ~isvector(col_sel)
                error('Row and column selections must be vectors');
            end
            
            row_sel = unique(row_sel);
            col_sel = unique(col_sel);
             
            obj.mask = false( obj.array_size );
            obj.mask(row_sel,col_sel) = true;     
            
            obj.net_size = [numel(row_sel) numel(col_sel)];
            obj.row_select = row_sel;
            obj.col_select = col_sel;
            % Initialize array connection:
            obj.array = dpe_writing();
            obj.array.connect();
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
            %   'mode' = 'slow' (TODO: add option for partial array reads)
            [~,conductances] = evalc('obj.array.read(varargin{:})'); % HARDWARE CALL
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
            
            V_gate = obj.expand(V_gate);
            V_in = obj.expand(V_in); % Voltage applied to each column ( Should be *V_set?)
            V_out = obj.expand(V_out); % Voltage applied to each row on a subsequent pulse
            
            % Only handles nonnegative voltages
            
            if any(V_in(:)>0) && any(V_out(:)>0)
                warning('Did you mean to set and reset all in one call?')
            end

            if any(V_in(:)>0)
                %obj.array.batch_set(V_in,V_gate,'print',false); % HARDWARE CALL
                [~] = evalc('obj.array.batch_set(V_in, V_gate)');
            end
            if any(V_out(:)>0)
                %obj.array.batch_reset(V_out, V_gate,'print',false); % HARDWARE CALL
                [~] = evalc('obj.array.batch_reset(V_out, V_gate)');
            end
        end
%%        
        function I_out = read_current(obj,V_in, varargin)
            % I_OUT = READ_CURRENT(V_IN) does matrix-matrix multiplication.
            % I_OUT = transpose(G)*V_IN when G is 128x64. Note that this
            % function interprets the orientations of I_OUT and V_in
            % differently than VMM_hardware.
            % V_IN can be either a row cell array of column vectors or a matrix.
            % In either case, I_OUT will be a matrix.
            % Optional args:
            %   gain = 1
            
            if iscell(V_in)
                try
                    cell2mat(V_in);
                catch
                    error('Unrecognized input format')
                end
            end
            
            % Original
            % temp = zeros(obj.net_size(1), size(V_in,2));
            % Corrected as
            temp = zeros(obj.array_size(1), size(V_in,2));
            temp(obj.row_select,:) = V_in;
            V_in = temp';
            
            %temp = padarray(V_in,obj.net_corner(1)-1,0,'pre'); % throwaway
            %V_in = padarray(temp,obj.array_size(1)-obj.net_corner(1)-obj.net_size(1)+1,0,'post');
            
            %[~,I_out] = evalc('obj.array.VMM_hardware(V_in, varargin{:})'); %HARDWARE CALL
            I_out = obj.array.VMM_hardware(V_in, varargin{:}); %HARDWARE CALL
            I_out = I_out(:,obj.col_select);
            I_out = I_out';
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
            
                        
            % Second part: Expand b to array_size
            if nargin<3
                x = 0;
            end
            
            c = zeros(obj.array_size)+x;
            c(obj.mask) = b;
        end
    end
end