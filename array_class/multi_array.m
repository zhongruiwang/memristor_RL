classdef multi_array < memristor_array
    % This class sits on top of another memristor_array, and holds a
    % collection of subarrays on that array that can be used together or
    % separately.
    % Useful for multilayer neural networks implemented on a single array.
    properties
        net_size
        array
        subs
    end
    methods
    %% Basic functions:
        %%
        function obj = multi_array(array)
            % OBJ = MULTI_ARRAY(ARRAY) creates a multi_array object on top
            % of the memristor_array ARRAY.
            % SUBS is initialized empty.
            obj.array = array;
            obj.net_size = array.net_size;
            obj.subs = {}; % Could replace subs with just a list of masks?
        end
        
        %%
        function conductances = read_conductance(obj, varargin)
            % G = OBJ.READ_CONDUCTANCE('NAME', VALUE) reads the
            % conductances of the underlying memristor_array.
            conductances = obj.array.read_conductance(varargin{:});
        end
        
        %%
        function update_conductance(obj, V_in, V_out, V_gate)
            % OBJ.UPDATE_CONDUCTANCE(V_IN, V_OUT, V_GATE) applies a single
            % update pulse to the underlying memristor_array.
            obj.array.update_conductance(V_in, V_out, V_gate)
        end
        
        %%
        function I_out = read_current(obj, V_in,varargin)
            % I_OUT = OBJ.READ_CURRENT(V_IN) reads the current from the
            % underlying memristor_array with input V_IN.
            I_out = obj.array.read_current(V_in,varargin{:});
        end
        
    %% Subarray functions:
        function add_sub(obj, net_corner, net_size)
            % OBJ.ADD_SUB(OBJ, NET_CORNER, NET_SIZE) creates a rectangular
            % subarray and saves it in OBJ.SUBS.
            
            % Subarrays are always ordered as they are added
            obj.subs{end+1} = subarray1(obj.array, net_corner, net_size); 
            
            % Check that they are nonoverlapping:
            check = zeros(obj.net_size);
            for i=1:length(obj.subs)
                check = check+obj.subs{i}.mask;
            end
            if any(check(:)>1)
                warning('Overlapping arrays created')
            end
        end
        
        %%
        function [G, fullG] = read_subs(obj,varargin)
            % G = OBJ.READ_SUBS('NAME', VALUE) reads each subarray and
            % returns its conductance matrix. G{I} is the conductance of
            % OBJ.SUBS{I}. 
            % [G, rawG] = OBJ.READ_SUBS(...) also returns the raw
            % conductance matrix for the full spanning array.
            % Optional arguments are the same as for the underlying array's
            % read operation.
            fullG = obj.array.read_conductance(varargin{:});
            
            %if isa(obj.array, 'real_array1')
            %    figure(200); clf;
            %    imagesc(fullG); % TEMPORARY
            %    title('G')
            %end
            
            G = cellfun(@(x) reshape(fullG(x.mask),x.net_size), obj.subs,'UniformOutput',false);
        end
        %%
        function pulse = update_subs(obj,V_in,V_out,V_gate)
            % OBJ.UPDATE_SUBS(V_IN,V_OUT,V_GATE) updates each subarray
            % according to the specified pulse parameters. 
            % PULSE = OBJ.UPDATE_SUBS(...) returns the expanded voltages,
            % as {V_in V_out V_gate}
            % Each voltage may be:
            %   A cell array of the same size as OBJ.SUBS
            %   A vector of the same size as OBJ.SUBS
            %   A scalar (the same value is applied to all subarrays)
            %   'GND'
            
            V_in = obj.expand(V_in);
            V_out = obj.expand(V_out);
            V_gate = obj.expand(V_gate);
            
            in = zeros(obj.net_size);
            out = zeros(obj.net_size);
            gate = zeros(obj.net_size);
            
            for i = 1:length(obj.subs)
                in = in + V_in{i};
                out = out + V_out{i};
                gate = gate + V_gate{i};
            end
            
            obj.array.update_conductance(in,out,gate);
            pulse = {in out gate};
        end
        %% Utilities
        function b = expand(obj,a)
            % B = OBJ.EXPAND(A) attempts to expand A to match each
            % subarray. A can be:
            %   A cell array of the same size as OBJ.SUBS
            %   A vector of the same size as OBJ.SUBS
            %   A scalar
            %   'GND'
            % B is always a cell array, with SIZE(B{I}) == OBJ.NET_SIZE for
            % all I and SIZE(B) == SIZE(OBJ.SUBS)
            
            b = cell(size(obj.subs));
            
            if iscell(a) && numel(a) == numel(obj.subs)
                for i = 1:length(obj.subs)
                    b{i} = obj.subs{i}.expand(a{i}); % Expand each entry
                end
            elseif strcmpi(a, 'GND') || (isscalar(a) && isnumeric(a))
                for i=1:length(obj.subs)
                    b{i} = obj.subs{i}.expand(a); % Expand a
                end
            elseif isvector(a) && numel(a) == numel(obj.subs)
                b = cell(size(a));
                for i = 1:length(obj.subs)
                    b{i} = obj.subs{i}.expand(a(i)); % Expand each element
                end
            else
                error('Not sure how to expand this input')
            end
        end
        %%
       function update_direct(obj,dG) 
           % Used only for debugging
           % Will delete eventually
           % dG must be a cell array, formatted to match each sub
           delta = zeros(obj.net_size);
           for i=1:length(obj.subs)
               m = obj.subs{i}.mask;
               delta(m) = delta(m)+dG{i}(:); % Right?
           end
           obj.array.conductances = obj.array.conductances + delta;
           obj.array.conductances(obj.array.conductances < 0) = 0;
       end
        %%       
       function im = show(obj,fig)
        % Plots the layout
           if nargin < 2
               fig = 1;
           end
           
           im = zeros(obj.net_size);
           for i=1:length(obj.subs)
               im(obj.subs{i}.mask) = im(obj.subs{i}.mask) + i;
           end
           figure(fig); clf;
           imagesc(im);
           ax = gca;
           ax.XTick = 0:8:obj.net_size(2);
           ax.YTick = 0:8:obj.net_size(1);
       end
    end
end