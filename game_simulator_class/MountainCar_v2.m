classdef MountainCar_v2 < handle
    % MountainCar enviroment
    
    properties    
        
        min_position = -1.2; % Allowed car position
        max_position = 0.6;
        goal_position = 0.5; % The winning position
        
        mass = 0.2; % mass of car (default 0.2)
        force = 0.2; % force of each push (default 0.2)
        friction = 0.5; % coefficient of friction
        deltaT = 0.1; % time step
        
        legal_action=[1 2 3]; % Allowed actions, [Left, Stop, Right]
              
        state_now; % Current state [ Position; Velocity ]
                
        rail_obj;
        car_obj;
    end
    
    methods
        %%
        function obj = MountainCar_v2()
            % Create the mountain car enviroment
        end
        %%    
        function [state_next, reward, done] = nextstate(obj, action)
            % State evolution with an action
            
            % Check if the given action is allowed
            if ~ismember(action, obj.legal_action)
                error('Illegle action');
            end
            
            % Read the current state
            x = obj.state_now(1); % Location
            xdot = obj.state_now(2); % Velocity
            
            % New Location and velocity
            newx = x + obj.deltaT * xdot;
            newxdot = xdot + obj.deltaT * (-9.8 * obj.mass * cos(3 * x)...
                + obj.force / obj.mass * (action - 2) - obj.friction * xdot);
            
            % Inelastic collision with the left bound
            if newx < -1.2
                newx = -1.2; newxdot = 0;
            end           
            
            % Return states
            state_next = [newx; newxdot];
            obj.state_now = state_next;
            
            % Check if the episode is finished
            if newx >= obj.goal_position
                done = 1;
                reward = 0;
            else
                done = 0;
                reward = -1;
            end
        end
        %%
        function state_new = reset_episode(obj)
            % Restart the episode
            
            % Initalization
            state_new = [0.2 * rand - 0.6; 0];
            
            % Update current state
            obj.state_now = state_new;
        end
        %%
        function Mountaincar_plot(obj, action, plot_axes)
            % Graphic illustration of the enviroment
            
            axes(plot_axes);
            
            % Draw the rail (first time only)
            if isempty(obj.rail_obj)
                rail_x = linspace(obj.min_position, obj.max_position, 100);
                rail_y = sin(3 * rail_x);
                obj.rail_obj=plot(rail_x, rail_y);
                hold on
            end
            
            % Draw the car                
            x1 = obj.state_now(1)-0.05; % Box left x
            x2 = x1 + 0.1; % Box right x
            y1 = sin(3*obj.state_now(1)); % Box lower y
            y2 = y1+0.1; % Box upper y
            
            if action == 1
                color = [1 0 0]*.9;
            elseif action == 3
                color = [0 0 1]*.9;
            else
                color = [1 1 1]*.9;
            end
                     
            if isempty(obj.car_obj)
                obj.car_obj=fill([x1 x2 x2 x1],[y1 y1 y2 y2],color); % Get the "car" object
                hold off
            else
                set(obj.car_obj,'xdata',[x1 x2 x2 x1],'ydata',[y1 y1 y2 y2],'FaceColor',color);
            end
            
            drawnow;
        end
    end
end
