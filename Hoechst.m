
filepaths = "path";  
results_table = table();
img = imread(filepaths);
if size(img, 3) == 3
    grayImg = rgb2gray(img);
else
    grayImg = img;
end
gaussianFilter = fspecial('gaussian', [5 5], 2); 
gaussianFilteredImg = imfilter(grayImg, gaussianFilter, 'same');
threshold_SOX9 = entropyThreshold(gaussianFilteredImg);
binaryImg = imbinarize(gaussianFilteredImg, threshold_SOX9);
labeledImage = bwlabel(binaryImg);
stats = regionprops(labeledImage, 'Area', 'Image', 'BoundingBox');
finalBinaryImg = false(size(binaryImg));
validAreas = [stats.Area];
validIdx = validAreas >= 5 & validAreas <= 20000;  
for k = find(validIdx)
    bb = round(stats(k).BoundingBox);
    x_start = bb(1);
    y_start = bb(2);
    x_end = x_start + bb(3) - 1;
    y_end = y_start + bb(4) - 1;
    finalBinaryImg(y_start:y_end, x_start:x_end) = finalBinaryImg(y_start:y_end, x_start:x_end) | stats(k).Image;
end

figure, imshow(finalBinaryImg), title('Final Binarized Image with Valid Objects');
sox9_positive_count = sum(validIdx);
imageArea = numel(grayImg);
cell_density = sox9_positive_count / imageArea;
total_positive_pixels = sum(finalBinaryImg(:));  
newRow = table(filepaths, sox9_positive_count, imageArea, cell_density, total_positive_pixels, ...
               'VariableNames', {'Filepath', 'Cell_Count', 'Image_Area', 'Density', 'Total_Positive_Pixels'});
results_table = [results_table; newRow];
disp(['Number of SOX9 Positive Cells: ', num2str(sox9_positive_count)]);
disp(['Total Image Area (pixels): ', num2str(imageArea)]);
disp(['Density of SOX9 Positive Cells: ', num2str(cell_density), ' cells per pixel']);
disp(['Total Positive Pixels: ', num2str(total_positive_pixels)]);
save('SOX9_analysis_results.mat', 'results_table');

function threshold = entropyThreshold(I)
    I = im2double(I); 
    [counts, ~] = imhist(I);
    p = counts / sum(counts);
    maxEntropy = -Inf;
    threshold = 0;

    for t = 1:length(p) - 1
        background = p(1:t);
        object = p(t + 1:end);
        background = background / sum(background);
        object = object / sum(object);
        backgroundEntropy = -nansum(background .* log2(background + eps));
        objectEntropy = -nansum(object .* log2(object + eps));
        totalEntropy = backgroundEntropy + objectEntropy;

        if totalEntropy > maxEntropy
            maxEntropy = totalEntropy;
            threshold = t;
        end
    end

    threshold = double(threshold) / 255; 
end
