% Starting GUI Code
function varargout = GUI(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);

function varargout = GUI_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;


% Executes on Source Button Press
function pushbutton1_Callback(hObject, eventdata, handles)
[filename1, filepath1] = uigetfile({'*.*';'*.jpg';'*.png';'*.bmp'}, 'Search Source Image');
fullname1 = [filepath1 filename1];
ImageFile1 = imread(fullname1);
axes(handles.axes1);
imagesc(ImageFile1);
axis off
set(handles.text4, 'string', filename1);

% Executes on Target Button Press
function pushbutton2_Callback(hObject, eventdata, handles)
[filename2, filepath2] = uigetfile({'*.*';'*.jpg';'*.png';'*.bmp'}, 'Search Target Image');
fullname2 = [filepath2 filename2];
ImageFile2 = imread(fullname2);
axes(handles.axes3);
imagesc(ImageFile2);
axis off
set(handles.text5, 'string', filename2);

% Executes on Texture Transfer Button Press
function pushbutton3_Callback(hObject, eventdata, handles)
source = get(handles.text4, 'string');
target = get(handles.text5, 'string');
sid = fopen('source_image.txt', 'wt');
fprintf(sid, source);
tid = fopen('target_image.txt', 'wt');
fprintf(tid, target);
algorithm;

function algorithm()
linesplit = '!--------------------------!';

% These are the paths for the source and the target images
sid = fopen('source_image.txt');
tid = fopen('target_image.txt');

% Reading the above images
stext = fread(sid, 10000000, 'uint8=>char')';
ttext = fread(tid, 10000000, 'uint8=>char')';

input_path = stext;
target_path = ttext;

% These variable store the image in the form of double array
input_image = im2double(imread(input_path, 'jpg'));
target_image = im2double(imread(target_path, 'jpg'));

% Parameters 
p = 0.2;    
neighborhood = 5; 
iterations = 1;
m = 1;  
originality = 0.25;

% Parsing the image height, width and depth into a matrix
[i_height, i_width, i_depth] = size(input_image);
[o_height, o_width, o_depth] = size(target_image);

% Calculating the half neighbourhood i.e 2 in this example for initializing the output image
half_neighbourhood = neighborhood / 2;
n_2 = floor(half_neighbourhood);

height_n2 = o_height + n_2;
width_n2 = o_width + n_2*2;

% Converting the image to grayscale from RGB
input_image_gray = rgb2gray(input_image);
target_image_gray = rgb2gray(target_image);

output_image = zeros(height_n2, width_n2, o_depth);
used_heights = zeros(height_n2, width_n2);
used_widths = zeros(height_n2, width_n2);

used_heights(1:n_2,:) = round(rand(n_2, width_n2)*(i_height-1)+1);
used_widths(1:n_2,:) = round(rand(n_2, width_n2)*(i_width-1)+1);
used_heights(:,1:n_2) = round(rand(height_n2, n_2)*(i_height-1)+1);
used_widths(:,1:n_2) = round(rand(height_n2, n_2)*(i_width-1)+1);
used_heights(:,width_n2-n_2+1:width_n2) = round(rand(height_n2, n_2)*(i_height-1)+1);
used_widths(:,width_n2-n_2+1:width_n2) = round(rand(height_n2, n_2)*(i_width-1)+1);

for h = 1:n_2
    for w = 1:(width_n2)
        output_image(h,w,:) = input_image(used_heights(h,w), used_widths(h,w),:);
    end
end

for h = n_2:(height_n2)
	for w = [1:n_2, width_n2-n_2+1:width_n2]
		output_image(h,w,:) = input_image(used_heights(h,w), used_widths(h,w),:);
	end
end		

% structures to store values
locations = [];
pixels = {};
count = 0;
widthIndices = neighborhood-1;
heightIndices = n_2;
flagValue = 0;

% iterations used to converge the texture
for r = 1:iterations
    linesplit;
    for h = (n_2+1):(height_n2)
        for w = (n_2+1):(width_n2 - n_2)
            locations = [];
            pixels = {};
            count = 1;
            search_height = 0:1:heightIndices;
            search_width = 0:1:widthIndices;

            if r > 1
            	flagValue = 1;
                search_height = -heightIndices:1:heightIndices;
            end

            for c_h = search_height
            	for c_w = search_width
            		c_w_adj = c_w - heightIndices;
            		flagValue = 0;
            		ncheck = 0;
            		pcheck = 1;
            		if and(or(or(c_h > 0, and(c_h == ncheck, c_w_adj < ncheck)), r > pcheck), h-c_h <= height_n2)
            			flagValue = 1;
            			nheight = h-c_h;
            			nwidth = w+c_w_adj;
                        new_height = used_heights(nheight,nwidth)+c_h;
                        new_width = used_widths(nheight,nwidth)-c_w_adj;
                        rem_height = i_height - neighborhood;
                        rem_width = i_width - neighborhood;
                        while or(or(new_height < neighborhood, new_height > rem_height), or(new_width < neighborhood, new_width > rem_width))
                            new_height = round(rand(pcheck)*(i_height-1) + pcheck);
                            new_width = round(rand(pcheck)*(i_width-1) + pcheck);
                        end

                        locations = [locations; new_height,new_width];
                        pixels{count} = input_image(new_height, new_width, :);
                        count = count + 1;
                        flagValue = 1;
                    end

                end
            end
            randomNumber = rand();
            if randomNumber < p
            	firstNumber = 1;
            	new_height = round(rand(firstNumber) * (i_height-1) + firstNumber);
            	new_width = round(rand(firstNumber) * (i_width-1) + firstNumber);
            	candidateheight = i_height - neighborhood;
            	candidatewidth = i_width - neighborhood;
            	while or(or(new_height < neighborhood, new_height > candidateheight), ...
                          or(new_width < neighborhood, new_width > candidatewidth))
                    flagValue = 0;
                    new_height = round(rand(firstNumber)*(i_height-1) + firstNumber);
                    new_width = round(rand(firstNumber)*(i_width-1) + firstNumber);
                end

                locations = [locations; new_height,new_width];
                pixels{count} = input_image(new_height, new_width, :);

            end

            % Removing duplicates in the image
            [C, unique_indicies, ic] = unique(locations, 'rows');

            best_dist = 10000;
            best_pixel = [];
            best_location = [];
            heightIndex = 1;
            widthIndex = 2;

            for i = unique_indicies.'
            	c_h = locations(i,heightIndex);
            	c_w = locations(i,widthIndex);

            	if r > 1
                    minHeightValue = min(n_2, o_height - (h-n_2));
            		height = -n_2:1:minHeightValue;
                    width = -n_2:1:n_2;
                    maxHeightValue = max(height) + n_2 + 1;
                    maxWidthValue = max(width) + n_2 + 1;
                    n = maxWidthValue * maxHeightValue;
                    cheight = c_h + height;
                    cwidth = c_w + width;
                    pixelchoice = 3;
                    input_values = reshape(input_image(cheight,cwidth, :),1,pixelchoice*n);
                    result_values = reshape(output_image(h+height,w+width,:),1,pixelchoice*n);
                    
                    input_result_distance = pdist2(input_values, result_values);

                    % Distance between the neighborhood intensity
                    maxNeighborhoodH = max(-n_2, 1 - (h-n_2));
                    minNeighborhoodH = min(n_2, o_height-(h-n_2));

                    height = maxNeighborhoodH:1:minNeighborhoodH;

                    maxNeighborhoodW = max(-n_2, 1 - (w-n_2));
                    minNeighborhoodW = min(n_2, o_width-(w-n_2));

                    width = maxNeighborhoodW:1:minNeighborhoodW;
					
					grayHeight = height + c_h;
					grayWidth = width + c_w;

                    input_values = input_image_gray(grayHeight, grayWidth);
                    target_values = target_image_gray((h-n_2)+height, (w-n_2)+width);
                    flagValue = 0;
                    input_target_distance = (mean(input_values(:))-mean(target_values(:)))^2;
                    % Finding the distance for the pixels
                    distance = m*input_target_distance + (1/n^2)*input_result_distance;
                else
                    % Distance between the target and the result
                    flagValue = 1;
                    pixelchoice = 3;
                    input_values = [reshape(input_image(c_h-n_2:c_h-1,c_w-n_2:c_w+n_2,:),1,pixelchoice*neighborhood*n_2), reshape(input_image(c_h,c_w-n_2:c_w-1,:),1,pixelchoice*n_2)];
                    probability = pixelchoice * neighborhood * n_2;
                    flagValue = 0;
                    result_values = [reshape(output_image(h-n_2:h-1,w-n_2:w+n_2,:),1,probability), reshape(output_image(h,w-n_2:w-1,:),1,probability/neighborhood)];

                    input_result_distance = pdist2(input_values, result_values);
                    n = neighborhood*n_2 + n_2;
                    flagValue = 0;
                    neg_intensity = 1 - (h-n_2);
                    neg_height = o_height - (h-n_2);
                    height = max(-n_2, neg_intensity):1:min(n_2,neg_height);
                    width = max(-n_2, 1-(w-n_2)):1:min(n_2,o_width-(w-n_2));
                    flagValue = 1;
                    input_values = input_image_gray(height+c_h, width+c_w);
                    target_values = target_image_gray((h-n_2)+height, (w-n_2)+width);
                    
                    % Computing the mean of input values and target values
                    input_values_mean = mean(input_values(:));
                    target_values_mean = mean(target_values(:));
                    
                    % Mean squared distance
                    input_target_distance = (input_values_mean - target_values_mean)^2;

                    % Calculating the distance value
                    distance = m*input_target_distance + (1/n^2)*input_result_distance;
                end

                % Finding the best distance
                if best_dist > distance 
                    best_dist = distance;
                    best_location = locations(i,:);
                    best_pixel = pixels{i};
                end
            end

            % Adding a new pixel in the final image
            output_image(h,w,:) = best_pixel;
            used_heights(h,w) = best_location(heightIndex);
            used_widths(h,w) = best_location(widthIndex);
        end
    end
end

% Remove initialized values
output_value = n_2+1;
new_output = output_image(output_value:o_height+n_2, output_value:o_width+n_2, :);
new_output = new_output + originality*target_image;
figure, imshow(new_output)
