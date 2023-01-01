function [imB] = ContrastEnhancement(im, k1, k2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Binarization of degraded document images 
% 
%  im:   the input document image
%  k1:   first tunning parameter   variation range [0 0.4]
%  k2:   second tunning parameter  variation range [0.7 1.5]
%  imB:  binarized document image
% 
%  Reference:
%  D. Lu, X. Huang, and L. Sui, �Binarization of degraded document images based on contrast enhancement�, 
%  Int. J. Doc. Anal. Recognit. IJDAR, vol. 21, no. 1�2, pp. 123�135, Jun. 2018, 
%  doi: 10.1007/s10032-018-0299-9.
%
%  implemented by: Msc. Eng. David Castro Pi�ol
%  email: davidpinyol91@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Grayscale conversion
if (size(im,3)>1)
    imgray = rgb2gray(im);
else
    imgray = im;
end

% Ensure dimmensions for 4 blocks
imgray = imresize(imgray,4*round(size(imgray)/4));

%% Area Partition

% computing Contrast Image
rows = size(imgray,1);
columns = size(imgray,2);
imPad = [imgray; zeros(1,columns)];
imPad = [imPad zeros(rows+1,1)];

Ch = zeros(size(imgray));
Cv = zeros(size(imgray));

for r = 1:rows    
    for c = 1:columns   
        % Horizontal contrast
        Ch(r,c) = abs(imPad(r,c)-imPad(r+1,c));        
    end
end

for r = 1:rows    
    for c = 1:columns    
        % Vertical contrast
        Cv(r,c) = abs(imPad(r,c)-imPad(r,c+1));        
    end
end

% Final Image contrast
C = zeros(size(imgray)); 
for r = 1:rows    
    for c = 1:columns        
        C(r,c) = max([Cv(r,c) Ch(r,c)]);
    end
end

% maximum contrast
CEntireMax = max(C(:));

% first size block division
blockSize4 = [round(rows/2) round(columns/2)];

% second size block division
blockSize16 = [round(rows/4) round(columns/4)];

fun = @(block_struct)max(block_struct.data(:));

% Maxium contrast vector in each A,B,C,D region of the image
C_abcdMax = blockproc(double(C),blockSize4,fun);

% Maximun contrast vector in regions AA, AB,... of the image
C_AAMax = blockproc(double(C),blockSize16,fun);

% Performing subdivision
fun = @(block_struct)...
    BlockProcessing(block_struct.data,CEntireMax,k1);

% First logic level of background 
C_logic1 = blockproc(double(C),blockSize4,fun); 

% Auxiliaries variables
C_logicAux = C_logic1(:);

% Obtainig the second level matrix logic specifing where are background areas
C_background = zeros(size(C_AAMax)); % one logic corresponds to background

% Will store contrast significant variance
C_signVariance = zeros(size(C_AAMax)); % one logic corresponds to significance variance

% Will store comparatively significant
C_compSignVar = zeros(size(C_AAMax)); 

for i = 1:length(C_logicAux)
    
    % performing second subdivision        
    if(not(C_logicAux(i)))
        
        % get the values of second division
        [CabMax,xcoor,ycoor] = GetValuesSecondDivision(i,C_AAMax);
        
        % No significant area, also background
        C_background(xcoor,ycoor) = CabMax <= k1*C_abcdMax(i);         
        
        % region with significant variance area
        C_signVariance(xcoor,ycoor) = CabMax >= k2*C_abcdMax(i);
        
        % comparatively significant area
        C_compSignVar(xcoor,ycoor) = ...
            and(CabMax <= k2*C_abcdMax(i),k1*C_abcdMax(i) <= CabMax );               
    end    
end

% ensuring logical values for further effective indexing
C_background = logical(C_background);
C_signVariance = logical(C_signVariance);
C_compSignVar = logical(C_compSignVar);

%% Grayscale contrast enhancement

Cenhanced = zeros(size(imgray));
imgray = double(imgray);

    %% For not-significant areas

    % getting each 16 block locations
    fun = @(block_struct)(block_struct.location);
    locations = blockproc(imgray,blockSize16,fun);
    locations = [locations(1:4,1:2); locations(1:4,3:4);...
        locations(1:4,5:6);locations(1:4,7:8)];

    % getting block sizes for each 16 blocks
    fun = @(block_struct)(block_struct.blockSize);
    blockSizes = blockproc(imgray, blockSize16, fun);
    blockSizes = [blockSizes(1:4,1:2); blockSizes(1:4,3:4);...
        blockSizes(1:4,5:6);blockSizes(1:4,7:8)];

    % selecting values associated 
    seleLocations = locations(C_background(:),:);
    seleBlockSizes = blockSizes(C_background(:),:);

    % assignig values to the BW output image
    for i = 1:size(seleLocations,1)

        xylocated = seleLocations(i,:); 
        xyIncrement = seleBlockSizes(i,:);
        
        xVect = xylocated(1):xylocated(1)+xyIncrement(1)-1;
        yVect = xylocated(2):xylocated(2)+xyIncrement(2)-1;
        
        % assigning background
        Cenhanced(xVect,yVect) = 255;        
    end
      
    %% For significant Areas-Weak contrast Enhancement 

    % selecting values associated 
    seleLocations = locations(C_compSignVar(:),:);
    seleBlockSizes = blockSizes(C_compSignVar(:),:);
    
    % operations for each block
    for i = 1:size(seleLocations,1)

        xylocated = seleLocations(i,:); 
        xyIncrement = seleBlockSizes(i,:);
        
        xVect = xylocated(1):xylocated(1)+xyIncrement(1)-1;
        yVect = xylocated(2):xylocated(2)+xyIncrement(2)-1;
        
        auxMatrix = (imgray(xVect,yVect) - min(imgray(:)))...
           ./(max(imgray(:)) - min(imgray(:)));       
            
        % n = number of gray levels modified
        n = length(auxMatrix(:));
        
        % weak contrast enhancement
        ff = (n-1).*auxMatrix;   
        
        % scaling data in the range [0 255]
        ff = 255*ff./max(ff(:));        
        Cenhanced(xVect,yVect) = ff;    
        
    end
    
    Cenhanced = uint8(Cenhanced);
      
    %% For comparatively significant areas-Strong contrast Enhancement 
    
    % selecting values associated 
    seleLocations = locations(C_signVariance(:),:);
    seleBlockSizes = blockSizes(C_signVariance(:),:);
    
    % operations for each block
    for i = 1:size(seleLocations,1)

        xylocated = seleLocations(i,:); 
        xyIncrement = seleBlockSizes(i,:);
        
        xVect = xylocated(1):xylocated(1)+xyIncrement(1)-1;
        yVect = xylocated(2):xylocated(2)+xyIncrement(2)-1;
        
        auxMatrix = (imgray(xVect,yVect) - min(imgray(:)))...
            ./(max(imgray(:)) - min(imgray(:)));
        
        n = length(auxMatrix(:));          
        % strong contrast enhancement
        ff = (n*n-1).*(auxMatrix.^2);           
        % scaling data in the range [0 255]
        ff = 255*ff./max(ff(:));            
        Cenhanced(xVect,yVect) = ff;     
        
    end
    Cenhanced = uint8(Cenhanced);
    
%% Local Threshold estimation-Global evaluation

[counts,binLocations] = imhist(Cenhanced);

counts1 = [counts(1:length(counts)/2);zeros(length(counts)/2,1)];
counts2 = [zeros(length(counts)/2,1);counts(length(counts)/2+1:end)];

[~,ff_foreground] = max(counts1);
[~,ff_background] = max(counts2);

ff_foreground = binLocations(ff_foreground);   
ff_background = binLocations(ff_background);

% computing the threshold
T = round((ff_foreground + ff_background)/2);

% binarization
g = Cenhanced>T;

% ensure same size of the input image
imB = imresize(g,[size(im,1) size(im,2)]);

%% Crucial Functions

    function [output] = BlockProcessing(block_data,CEntireMax,k1)
    output = max(block_data(:))<=k1*CEntireMax;
end

function [values,xcoor,ycoor] = GetValuesSecondDivision(i,SecondMatrix)
%   i: goes from 1 to 4
%   SecondMatrix: secondDivision 4x4 matrix
%   values: Values in the region matrix 2x2

switch i
    case 1
        % A block 
        xcoor = 1:2;
        ycoor = 1:2;
        values = SecondMatrix(xcoor,ycoor);
    case 2
        % B block
        xcoor = 3:4;
        ycoor = 1:2;
        values = SecondMatrix(xcoor,ycoor);
    case 3
        % C block
        xcoor = 1:2;
        ycoor = 3:4;
        values = SecondMatrix(xcoor,ycoor);
    case 4
        % D block
        xcoor = 3:4;
        ycoor = 3:4;
        values = SecondMatrix(xcoor,ycoor);
    otherwise
        error('Unexpected input value');
end

end

end

