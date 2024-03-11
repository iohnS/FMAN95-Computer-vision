function [result]=pflat(data)
%PFLAT divides homogeneous coordinates by their last entry

last_coord = data(end,:);       
result = data;

for i = 1:length(data(1,:)) 
    result(:,i)=data(:,i)./last_coord(i); 
end    

end