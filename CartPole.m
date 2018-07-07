classdef CartPole < handle
    % CartPole enviroment
    
    properties
        gravity = 9.8; % Gravitational acceleration
        masscart = 1.0; % Cart mass
        masspole = 0.1; % Pole mass
        %total_mass = masspole + masscart; % Total mass
        total_mass = 1.1;
        polelength = 0.5 % Actually half the pole's length
        % polemass_length = masspole * polelength; % No idea 
        polemass_length = 0.05;
        force_push_mag = 10.0; % Magnitude of the force_push
        tau = 0.02; % Seconds between state updates
        
        theta_threshold_radians = 12 * 2 * pi / 360; % Angle of fail
        x_threshold = 2.4; % X-axis bounds
        
        legal_action=[-1 1]; % Values of legal actions.
        
        state_now; %The state [x x' theta theta']';
        done_flag=false; % The flag marks the current episode is finished or not
        
        foundation_plot;
        cart_plot;
        pole_plot;
    end
    
    methods
        function obj = CartPole()
        end
        
        function [state_next,reward,done]=nextstate(obj,action)
            % State evolution with the given action
            
            % Check if the given action is legal
            if ~ismember(action,obj.legal_action)
                error('Illegle action');
            end
            
            % Import the current state
            x=obj.state_now(1);
            x_dot=obj.state_now(2);
            theta=obj.state_now(3);
            theta_dot=obj.state_now(4);
            
            % Assign force_push based on action
            if action==obj.legal_action(2)
                force_push=obj.force_push_mag;
            else
                force_push=-obj.force_push_mag;
            end
            
            % Physics          
            temp=(force_push+obj.polemass_length*theta_dot^2*sin(theta))/obj.total_mass;
            thetaacc=(obj.gravity*sin(theta)-cos(theta)*temp)/...
                (obj.polelength*(4/3-obj.masspole*cos(theta)^2/obj.total_mass));
            xacc=temp-obj.polemass_length*thetaacc*cos(theta)/obj.total_mass;
            
            x=x+obj.tau*x_dot;
            x_dot=x_dot+obj.tau*xacc;
            theta=theta+obj.tau*theta_dot;
            % -------------------------------------------------------------
            % Note a change here to make the theta spans a similar range
            % with others (output only)
            theta_x10=theta*10;
            % -------------------------------------------------------------
            theta_dot=theta_dot+obj.tau*thetaacc;
            
            state_next=[x,x_dot,theta,theta_dot]';
            obj.state_now=state_next;
            
            % Check if the episode is finished
            done=any([x<-obj.x_threshold, x>obj.x_threshold,...
                theta<-obj.theta_threshold_radians, theta>obj.theta_threshold_radians]);
            
            if ~done
                % If survive for one more time step, get 1 reward
                reward=1;
            elseif ~obj.done_flag
                % In case the done_flag is not marked, so just meet done
                % condition
                obj.done_flag=true;
                reward=1;
            else
                % Already done
                warning('Episode is over. Reset to start a new game.');
                reward=0;
            end
            
            % -------------------------------------------------------------
            % For output only (internal state is still "theta")
            state_next=[x,x_dot,theta_x10,theta_dot]';
            % -------------------------------------------------------------
        end

        function state_new=reset_episode(obj)
            % Restart the episode
            
            % Initalization 1: Random x and theta. x' and theta' are 0.
            % state_new=[obj.x_threshold*(2*rand-1),0,obj.theta_threshold_radians*(2*rand-1),0]';
            % Initalization 2: based on cartpole.py
            state_new=0.1*(rand(4,1)-0.5);
            
            % -------------------------------------------------------------
            % Note state_new is the output version
            % state_new(3) or theta is x10... 
            % -------------------------------------------------------------
            
            % Update current state and done_flag
            obj.state_now=state_new;
            
            % -------------------------------------------------------------
            % Change the "internal" state(3) back to theta x1 (from x10)
            obj.state_now(3)=state_new(3)/10;
            % -------------------------------------------------------------
            obj.done_flag=false;
        end
            
        function CartPole_plot(obj, plot_axis_obj, extrainfo)
            % Plot the graphic illustration
            
            % Settings
            foundation_thick=0.2;
            cart_width=0.4;
            cart_height=0.2;
            pole_width=0.1;
            pole_height=1;
            
            % The foundation coordinates
            if isempty(obj.foundation_plot)
                x_foundation=[-obj.x_threshold, obj.x_threshold, obj.x_threshold, -obj.x_threshold];
                y_foundation=[0, 0, -foundation_thick, -foundation_thick];
            end
            
            % The cart coordinates
            x=obj.state_now(1);
            x_cart=[x-cart_width/2, x+cart_width/2, x+cart_width/2, x-cart_width/2];
            y_cart=[cart_height, cart_height, 0, 0];
                       
            % The pole coordinates
            theta=obj.state_now(3);
            x_pole=[pole_height*sin(theta)-0.5*pole_width*cos(theta), pole_height*sin(theta)+0.5*pole_width*cos(theta),...
                0.5*pole_width*cos(theta), -0.5*pole_width*cos(theta)];
            x_pole=x_pole+x;
            y_pole=[pole_height*cos(theta)+0.5*pole_width*sin(theta), pole_height*cos(theta)-0.5*pole_width*sin(theta),...
                -0.5*pole_width*sin(theta), 0.5*pole_width*sin(theta)];
            y_pole=y_pole+0.5*cart_height;
            
            axes(plot_axis_obj);
            
            if isempty(obj.foundation_plot)
                
                % First plot    
                hold on
                obj.foundation_plot=fill(x_foundation,y_foundation,[0.5 0.5 0.5]);
                obj.cart_plot=fill(x_cart,y_cart,[1 0 0]);
                obj.pole_plot=fill(x_pole,y_pole,[0 1 0]);
                hold off
                axis([-2.6 2.6 -0.5 1.4]);
            else
                set(obj.cart_plot,'xdata', x_cart,'ydata', y_cart);
                set(obj.pole_plot,'xdata', x_pole,'ydata', y_pole);
            end
            
            % Title
            title(['CartPole overall step #', num2str(extrainfo)]);
            
            % Draw
            drawnow;

        end
    end
end

