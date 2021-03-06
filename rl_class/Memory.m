classdef Memory < handle
    % Memory (of the agent)
    
    properties
        capacity
        samples        
        pointer = 0; % Point to the last element of the "samples"
    end
    
    methods
        %%
        function obj = Memory(capacity)
            % Create memory obj
            
            % Size of the memory
            obj.capacity = capacity;
            
            % Initial memory with the given size
            obj.samples = cell(capacity, 4);
        end
        %%
        function add_memory(obj, new_sample)
            % Add {state, action, reward, next_state} 1x4 cell to memory
                    
            if obj.pointer < obj.capacity
                % In case the memory is not full
                obj.pointer = obj.pointer + 1;
                obj.samples(obj.pointer, 1:4) = new_sample;
            else
                % In case memory is full, remove the oldest
                obj.samples(1, :) = [];
                obj.samples(obj.pointer, 1:4) = new_sample;
            end
        end
        %%
        function samples_recalled = sample_memory(obj, n)
            % Sample the experience
            
            % Choose the sample sizes
            n = min(n, obj.pointer);
            
            % Recall n different samples from the existing memory
            samples_index = sort(datasample(1:obj.pointer, n, 'Replace', false));
            
            % The recalled experience, cell array nx4 size.
            samples_recalled = obj.samples(samples_index, 1:4);
        end           
    end
end