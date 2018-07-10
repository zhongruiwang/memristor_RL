classdef MountainCar_v2 < handle
    % MountainCar Enviroment Open AI Gym
    
    properties    
        % Constants
        min_position = -1.2;
        max_position = 0.6;
        goal_position = 0.5;
        
        mass = 0.2;				% mass of car (default 0.2)
        force = 0.2;			% force of each push (default 0.2)
        friction = 0.5;			% coefficient of friction (default 0.5)
        deltaT = 0.1;			% time step for integration (default 0.1)
        
        legal_action=[1 2 3]; % [ Left, Stop, Right]
        
        % Record
        state_now; % [ Position; Velocity ]
        
        % For animation use only
        rail_obj; 
        car_obj;
    end
    
    methods
        %%
        function obj=MountainCar_v2()
        end
        %%    
        function [state_next,reward,done]=nextstate(obj,action)
            % State evolution with the given action
            
            % Check if the given action is legal
            if ~ismember(action,obj.legal_action)
                error('Illegle action');
            end
            
            % Import the current state            
            x = obj.state_now(1); % Location
            xdot = obj.state_now(2); % Velocity
            
            % New Location and velocity
            newx = x + obj.deltaT * xdot;
            newxdot = xdot + obj.deltaT * (-9.8 * obj.mass * cos(3 * x)...
                + obj.force / obj.mass * (action-2) - obj.friction * xdot);
            
            % Break left bound case (assume 0 velocity)
            if newx < -1.2
                newx = -1.2; newxdot = 0;
            end           
            
            % Return states
            state_next=[newx;newxdot];
            obj.state_now=state_next;
            
            % Check if the episode is finished
            if newx >= obj.goal_position
                done=1;
                reward=0;
            else
                done=0;
                reward=-1;
            end
        end
        %%
        function state_new=reset_episode(obj)
            % Restart the episode
            
            % Initalization : based on cartpole.py
            state_new=[0.2*rand-0.6;0];
            
            % Update current state
            obj.state_now=state_new;
        end
        %%
        function Mountaincar_plot(obj, action, plot_axes)
            % Draw the new "car" and "action color"
            
            axes(plot_axes);
            
            % Draw the rail (first time only)
%            if isempty(obj.rail_obj)
                rail_x = linspace(obj.min_position, obj.max_position, 100);
                rail_y = sin(3 * rail_x);
                obj.rail_obj=plot(rail_x, rail_y);
                hold on
%            end
            
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
                     
%            if isempty(obj.car_obj)
                obj.car_obj=fill([x1 x2 x2 x1],[y1 y1 y2 y2],color); % Get the "car" object
                hold off
%            else
%                set(obj.car_obj,'xdata',[x1 x2 x2 x1],'ydata',[y1 y1 y2 y2],'FaceColor',color);
%            end
            
            drawnow;
        end
    end
end
